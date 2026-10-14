from pathlib import Path

from langchain_core.documents import Document

from rag_core import inventory_signature, split_documents


def test_split_documents_preserves_source_and_page_metadata():
    page = Document(
        page_content=("Eligibility requires a residence certificate. " * 30),
        metadata={"source": "scheme_1.pdf", "page_number": 4, "page": 4},
    )
    chunks = split_documents([page], chunk_size=300, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "scheme_1.pdf" for c in chunks)
    assert all(c.metadata["page_number"] == 4 for c in chunks)
    assert all(c.page_content for c in chunks)


def test_inventory_signature_changes_when_file_changes(tmp_path: Path):
    pdf = tmp_path / "scheme.pdf"
    pdf.write_bytes(b"first")
    sig1 = inventory_signature([pdf])
    pdf.write_bytes(b"second content")
    sig2 = inventory_signature([pdf])
    assert sig1 != sig2


def test_inventory_signature_changes_for_chunk_settings(tmp_path: Path):
    pdf = tmp_path / "scheme.pdf"
    pdf.write_bytes(b"pdf-ish bytes")
    sig1 = inventory_signature([pdf], chunk_size=1000, chunk_overlap=160)
    sig2 = inventory_signature([pdf], chunk_size=800, chunk_overlap=100)
    assert sig1 != sig2


def test_inventory_signature_changes_for_embedding_model(tmp_path: Path):
    pdf = tmp_path / "scheme.pdf"
    pdf.write_bytes(b"pdf-ish bytes")
    sig_multilingual = inventory_signature(
        [pdf], embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    sig_english = inventory_signature(
        [pdf], embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    assert sig_multilingual != sig_english
