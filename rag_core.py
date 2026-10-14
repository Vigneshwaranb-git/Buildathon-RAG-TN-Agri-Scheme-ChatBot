"""Core retrieval and indexing functions for the Tamil Nadu Scheme RAG app."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 160
MANIFEST_NAME = "manifest.json"


def discover_pdfs(pdf_dir: str | Path) -> list[Path]:
    """Recursively discover PDF files in a directory, in stable order."""
    root = Path(pdf_dir).expanduser().resolve()
    if not root.exists():
        return []
    if not root.is_dir():
        raise NotADirectoryError(f"PDF path is not a directory: {root}")
    return sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"), key=lambda p: str(p).lower())


def inventory_signature(
    pdf_paths: Iterable[Path],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> str:
    """Fast corpus/config fingerprint for checking whether a saved index is stale."""
    rows: list[dict[str, object]] = []
    for path in sorted((Path(p).resolve() for p in pdf_paths), key=lambda p: str(p).lower()):
        stat = path.stat()
        rows.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    payload = {
        "files": rows,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "schema_version": 2,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def make_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Create local Hugging Face sentence-transformer embeddings.

    The model is downloaded from Hugging Face on first use and cached locally.
    PDF text is embedded on the user's machine; no OpenAI key is needed here.
    """
    if not model.strip():
        raise ValueError("Choose a Hugging Face sentence-transformer model.")
    return HuggingFaceEmbeddings(
        model_name=model.strip(),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        show_progress=True,
    )


def load_pdf_pages(pdf_paths: Iterable[Path]) -> tuple[list[Document], list[str], list[str]]:
    """Read each page as a Document; normalize citation metadata to 1-based pages."""
    pages: list[Document] = []
    warnings: list[str] = []
    failed: list[str] = []
    paths = list(pdf_paths)
    if paths:
        try:
            common_root = Path(os.path.commonpath([str(p.resolve()) for p in paths]))
        except ValueError:
            # Defensive fallback for files located on different Windows drives.
            common_root = Path(paths[0]).resolve().parent
        if common_root.is_file():
            common_root = common_root.parent
    else:
        common_root = Path.cwd()

    for pdf_path in paths:
        try:
            loaded_pages = PyPDFLoader(str(pdf_path)).load()
            if not loaded_pages:
                warnings.append(f"{pdf_path.name}: PDF-இல் இருந்து பக்கங்களைப் பெற முடியவில்லை.")
                continue
            try:
                display_source = str(pdf_path.resolve().relative_to(common_root))
            except ValueError:
                display_source = pdf_path.name
            if display_source in {"", "."}:
                display_source = pdf_path.name
            usable_for_file = 0
            for index, page_doc in enumerate(loaded_pages):
                content = (page_doc.page_content or "").strip()
                page_raw = page_doc.metadata.get("page", index)
                try:
                    page_number = int(page_raw) + 1
                except (TypeError, ValueError):
                    page_number = index + 1
                if not content:
                    warnings.append(f"{display_source}, பக்கம் {page_number}: தேர்ந்தெடுக்கக்கூடிய உரை இல்லை (ஸ்கேன் செய்யப்பட்ட படமாக இருக்கலாம்).")
                    continue
                metadata = {
                    "source": display_source,
                    "file_name": pdf_path.name,
                    "page_number": page_number,
                    "page": page_number,
                }
                pages.append(Document(page_content=content, metadata=metadata))
                usable_for_file += 1
            if usable_for_file == 0:
                warnings.append(f"{display_source}: தேடக்கூடிய உரை இல்லை; இந்த PDF-க்கு OCR தேவைப்படலாம்.")
        except Exception as exc:  # Keep indexing other valid files if one PDF is malformed.
            failed.append(f"{pdf_path.name}: PDF-ஐப் படிக்க முடியவில்லை ({type(exc).__name__}: {exc})")
    return pages, warnings, failed


def split_documents(pages: list[Document], chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[Document]:
    if chunk_size < 200:
        raise ValueError("உரைத்துண்டு அளவு குறைந்தது 200 எழுத்துகளாக இருக்க வேண்டும்.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("உரைத்துண்டு மேலொட்டல் பூஜ்யம் அல்லது அதற்கு மேல் இருக்க வேண்டும்; மேலும் அது chunk அளவைவிடக் குறைவாக இருக்க வேண்டும்.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["page_number"] = int(chunk.metadata.get("page_number", chunk.metadata.get("page", 1)))
        chunk.metadata["source"] = str(chunk.metadata.get("source", "Unknown PDF"))
    return chunks


def read_manifest(index_dir: str | Path) -> dict | None:
    path = Path(index_dir) / MANIFEST_NAME
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def index_is_current(index_dir: str | Path, signature: str) -> bool:
    directory = Path(index_dir)
    manifest = read_manifest(directory)
    return bool(
        manifest
        and manifest.get("inventory_signature") == signature
        and (directory / "index.faiss").is_file()
        and (directory / "index.pkl").is_file()
    )


def _atomic_write_json(path: Path, data: dict) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix="manifest_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def build_index(
    pdf_paths: list[Path],
    index_dir: str | Path,
    embeddings: Embeddings,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Build a FAISS vector index and replace any previous local index safely."""
    if not pdf_paths:
        raise ValueError("தேர்ந்தெடுத்த கோப்புறையில் PDF ஆவணங்கள் எதுவும் இல்லை.")
    target = Path(index_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"{len(pdf_paths)} PDF ஆவணங்களில் இருந்து உரையைப் படிக்கிறது...")
    pages, warnings, failed = load_pdf_pages(pdf_paths)
    if not pages:
        details = "\n".join(failed[:5])
        raise ValueError("PDF ஆவணங்களில் தேடக்கூடிய உரை கிடைக்கவில்லை. ஸ்கேன் செய்யப்பட்ட PDF-களுக்கு குறியீட்டுக்கு முன் OCR தேவை." + (f" விவரம்: {details}" if details else ""))
    if progress:
        progress(f"உரை கிடைத்த {len(pages)} பக்கங்கள் கண்டறியப்பட்டன; உரைத்துண்டுகளாகப் பிரிக்கிறது...")
    chunks = split_documents(pages, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("கொடுக்கப்பட்ட PDF-களில் இருந்து உரைத்துண்டுகள் உருவாக்கப்படவில்லை.")
    if progress:
        progress(f"{len(chunks)} உரைத்துண்டுகளுக்கான embedding உருவாக்கி FAISS அறிவுத்தளத்தை உருவாக்குகிறது...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    signature = inventory_signature(pdf_paths, embedding_model, chunk_size, chunk_overlap)
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_signature": signature,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "pdf_count": len(pdf_paths),
        "pages_with_text": len(pages),
        "chunk_count": len(chunks),
        "source_files": [str(p.resolve()) for p in pdf_paths],
        "warnings": warnings,
        "failed_files": failed,
    }
    staging = target.with_name(target.name + ".staging")
    backup = target.with_name(target.name + ".backup")
    for directory in (staging, backup):
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
    try:
        vector_store.save_local(str(staging))
        _atomic_write_json(staging / MANIFEST_NAME, manifest)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    if progress:
        progress("FAISS அறிவுத்தளம் வெற்றிகரமாகச் சேமிக்கப்பட்டது.")
    return {**manifest, "warnings": warnings, "failed_files": failed}


def load_index(index_dir: str | Path, embeddings: Embeddings) -> FAISS:
    """Load only indexes created locally by this application.

    LangChain's FAISS persistence includes a pickle file. Loading it can execute code,
    so only load this directory if it is under your control and was created by this app.
    """
    return FAISS.load_local(
        str(Path(index_dir).resolve()),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def extract_message_text(message_content: object) -> str:
    """Normalize ChatOpenAI message content across string/block return forms."""
    if isinstance(message_content, str):
        return message_content.strip()
    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    return str(message_content).strip()
