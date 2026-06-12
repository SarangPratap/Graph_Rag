"""Integration test: load PDFs, chunk, run entity extraction on first 3 chunks."""

from __future__ import annotations

from graphrag.ingestion.chunker import chunk_documents
from graphrag.ingestion.pdf_loader import load_pdfs_from_folder
from graphrag.extraction.entity_extractor import extract


def main() -> None:
    """Load sample PDFs, chunk them, extract entities from first 3 chunks."""
    docs = load_pdfs_from_folder("./sample_pdfs")
    print(f"\nLoaded {len(docs)} document(s).\n")

    chunks = chunk_documents(docs)
    print(f"Total chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:3], 1):
        print(f"--- Chunk {i} (id={chunk.id[:8]}...) ---")
        result = extract(chunk)
        first = result.entities[0] if result.entities else None
        print(f"  Entities : {len(result.entities)}")
        print(f"  Relations: {len(result.relations)}")
        if first:
            print(f"  First entity: name={first.name!r}, type={first.type}")
        print()


if __name__ == "__main__":
    main()
