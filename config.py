"""Application configuration loaded from the project .env file.

Environment variables provided by the host take precedence over .env values,
which is the recommended behaviour for deployed applications.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
ENV_FILE = APP_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=False)


def _path_from_env(name: str, default: str) -> Path:
    value = os.getenv(name, default).strip().strip('"').strip("'")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path.resolve()


def _int_from_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f".env அமைப்பு {name} முழு எண்ணாக இருக்க வேண்டும்; கிடைத்தது {raw!r}.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f".env அமைப்பு {name} மதிப்பு {minimum} முதல் {maximum} வரை இருக்க வேண்டும்; கிடைத்தது {value}.")
    return value


def _key_from_env() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    if not key or key.lower() in {
        "replace_with_your_openai_api_key",
        "your_openai_api_key",
        "your_actual_api_key",
        "replace-me",
    }:
        return ""
    return key


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_chat_model: str
    hf_embedding_model: str
    pdf_folder: Path
    rag_index_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_upload_mb: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        chunk_size = _int_from_env("CHUNK_SIZE", 1000, 200, 5000)
        chunk_overlap = _int_from_env("CHUNK_OVERLAP", 160, 0, 4999)
        if chunk_overlap >= chunk_size:
            raise ValueError(".env கோப்பில் CHUNK_OVERLAP மதிப்பு CHUNK_SIZE-ஐவிடக் குறைவாக இருக்க வேண்டும்.")
        return cls(
            openai_api_key=_key_from_env(),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
            hf_embedding_model=os.getenv(
                "HF_EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ).strip() or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            pdf_folder=_path_from_env("PDF_FOLDER", "scheme_pdfs"),
            rag_index_dir=_path_from_env("RAG_INDEX_DIR", ".rag_index"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=_int_from_env("TOP_K", 6, 1, 20),
            max_upload_mb=_int_from_env("MAX_UPLOAD_MB", 50, 1, 500),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        )


settings = Settings.from_env()
