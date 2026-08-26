"""Load PDF, DOCX, TXT, Markdown, JSON, and Excel files into Documents."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import structlog

from graphrag.core.content_hash import compute_content_hash
from graphrag.core.models import Document

log = structlog.get_logger(__name__)


def load_document(file_path: str | Path) -> Document:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    text = load_document_content(path.name, path.read_bytes())

    doc = Document(
        filename=path.name,
        source_path=str(path.resolve()),
        raw_text=text,
        metadata={"extension": suffix, "size_bytes": path.stat().st_size},
        # Computed here, at the single point every format converges to str,
        # so the whole pipeline downstream can compare "is this the same
        # document as last run?" without re-reading the file.
        content_hash=compute_content_hash(text),
    )
    log.info("document_loader.loaded", filename=doc.filename, chars=len(text))
    return doc


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _load_docx(path: Path) -> str:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    return "\n".join(para.text for para in doc.paragraphs)


def load_document_content(filename: str, content: bytes) -> str:
    """Extract text from a local or remotely downloaded document payload."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
    if suffix == ".docx":
        from docx import Document as DocxDocument
        return "\n".join(para.text for para in DocxDocument(BytesIO(content)).paragraphs)
    if suffix == ".json":
        return json.dumps(json.loads(content.decode("utf-8")), ensure_ascii=False, indent=2, sort_keys=True)
    if suffix == ".xlsx":
        from openpyxl import load_workbook
        workbook = load_workbook(BytesIO(content), read_only=True, data_only=True)
        rows: list[str] = []
        for sheet in workbook.worksheets:
            rows.append(f"# Sheet: {sheet.title}")
            rows.extend(" | ".join("" if value is None else str(value) for value in row)
                        for row in sheet.iter_rows(values_only=True))
        return "\n".join(rows)
    if suffix in {".txt", ".md", ".csv"}:
        return content.decode("utf-8")
    raise ValueError(f"Unsupported file type: {suffix}")
