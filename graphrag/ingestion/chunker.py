"""Semantic text chunker — uses Docling HybridChunker when available, paragraph splitter as fallback."""

from __future__ import annotations

import logging
import re

from graphrag.models import Chunk, Document

logger = logging.getLogger(__name__)

MAX_CHARS = 1600   # fallback paragraph chunker — ~400 tokens at 4 chars/token
MIN_PARA_CHARS = 20

# HybridChunker is initialised lazily on first use to avoid import cost at module load.
_hybrid_chunker = None


def _get_hybrid_chunker():
    """Return a shared HybridChunker instance, creating it on first call."""
    global _hybrid_chunker
    if _hybrid_chunker is None:
        from docling.chunking import HybridChunker
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _hybrid_chunker = HybridChunker(tokenizer=tokenizer, max_tokens=100, merge_peers=True)
        logger.info("HybridChunker initialised with max_tokens=100")
    return _hybrid_chunker


def chunk_document(doc: Document) -> list[Chunk]:
    """Split a Document into Chunks, using HybridChunker if available.

    If the DoclingDocument for this source is cached (i.e. load_pdf was called first),
    HybridChunker is used for semantic ~100-token chunks. Otherwise falls back to the
    paragraph-boundary splitter.
    """
    from graphrag.ingestion.pdf_loader import get_docling_doc
    docling_doc = get_docling_doc(doc.source)
    if docling_doc is not None:
        chunks = _chunk_with_hybrid(doc, docling_doc)
        print(f"HybridChunker: {doc.title} -> {len(chunks)} chunks")
        return chunks
    chunks = _chunk_with_paragraphs(doc)
    print(f"ParagraphChunker: {doc.title} -> {len(chunks)} chunks")
    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    """Chunk every Document and return a single flat list of all Chunks."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


def _chunk_with_hybrid(doc: Document, docling_doc) -> list[Chunk]:
    """Use Docling HybridChunker to produce semantically bounded ~100-token chunks."""
    chunker = _get_hybrid_chunker()
    chunks: list[Chunk] = []
    for i, dl_chunk in enumerate(chunker.chunk(docling_doc)):
        text = dl_chunk.text.strip()
        if not text or len(text) < MIN_PARA_CHARS:
            continue
        chunks.append(Chunk(
            text=text,
            source=doc.source,
            doc_title=doc.title,
            doc_id=doc.id,
            position=i,
        ))
    return chunks


def _chunk_with_paragraphs(doc: Document) -> list[Chunk]:
    """Fallback: split at paragraph boundaries and merge up to MAX_CHARS."""
    raw_paras = re.split(r"\n{2,}", doc.text)
    paras = [p.strip() for p in raw_paras if len(p.strip()) >= MIN_PARA_CHARS]

    units: list[str] = []
    for para in paras:
        if len(para) > MAX_CHARS:
            units.extend(_split_at_sentences(para))
        else:
            units.append(para)

    return [
        Chunk(
            text=text,
            source=doc.source,
            doc_title=doc.title,
            doc_id=doc.id,
            position=i,
        )
        for i, text in enumerate(_merge_units(units))
    ]


def _split_at_sentences(text: str) -> list[str]:
    """Split a long paragraph into sentence-boundary segments under MAX_CHARS."""
    sentences = text.split(". ")
    units: list[str] = []
    current = ""
    for i, sent in enumerate(sentences):
        piece = sent if i == len(sentences) - 1 else sent + ". "
        if not current:
            current = piece
        elif len(current) + len(piece) <= MAX_CHARS:
            current += piece
        else:
            if current.strip():
                units.append(current.strip())
            current = piece
    if current.strip():
        units.append(current.strip())
    return units


def _merge_units(units: list[str]) -> list[str]:
    """Greedily merge adjacent units into chunks up to MAX_CHARS."""
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + 2 + len(unit) <= MAX_CHARS:
            current = current + "\n\n" + unit
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks
