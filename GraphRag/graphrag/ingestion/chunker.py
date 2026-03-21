"""Semantic chunking using LangChain's RecursiveCharacterTextSplitter."""

from __future__ import annotations

from graphrag.core.config import get_settings
from graphrag.core.models import Chunk, Document


def chunk_document(document: Document) -> list[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    cfg = get_settings().ingestion
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.get("chunk_size", 512),
        chunk_overlap=cfg.get("chunk_overlap", 64),
        length_function=len,
    )

    texts = splitter.split_text(document.raw_text)
    return [
        Chunk(
            document_id=document.id,
            text=text,
            chunk_index=i,
        )
        for i, text in enumerate(texts)
    ]
