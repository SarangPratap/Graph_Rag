"""Load PDFs from a folder using Docling and return Document model instances."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from docling.document_converter import DocumentConverter
from dotenv import load_dotenv

from graphrag.models import Document

load_dotenv()

logger = logging.getLogger(__name__)

_converter = DocumentConverter()

# Cache of source_path → DoclingDocument so chunker can use HybridChunker
# without re-loading the PDF.
_docling_cache: dict[str, Any] = {}


def get_docling_doc(source: str) -> Optional[Any]:
    """Return the cached DoclingDocument for this source path, or None."""
    return _docling_cache.get(source)


def load_pdf(file_path: str) -> Document:
    """Parse a single PDF with Docling and return a Document model instance.

    The DoclingDocument is cached so the chunker can use HybridChunker without
    re-loading the file.
    """
    path = Path(file_path).resolve()
    result = _converter.convert(str(path))
    _docling_cache[str(path)] = result.document
    text = result.document.export_to_markdown()
    pages = len(result.document.pages)
    return Document(
        title=path.stem,
        source=str(path),
        text=text,
        pages=pages,
    )


def load_pdfs_from_folder(folder: str) -> list[Document]:
    """Load every PDF in a folder and return a list of Document instances.

    Files that fail to parse are skipped; the error is logged and loading
    continues so a single bad file never aborts the whole batch.
    """
    folder_path = Path(folder).resolve()
    pdf_files = sorted(folder_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", folder_path)
        return []

    docs: list[Document] = []
    for pdf in pdf_files:
        print(f"Loading {pdf.name}...")
        try:
            docs.append(load_pdf(str(pdf)))
        except Exception as exc:
            logger.error("Skipping %s — %s: %s", pdf.name, type(exc).__name__, exc)

    return docs
