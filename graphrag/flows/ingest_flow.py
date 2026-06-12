"""Prefect flow: ingest a folder of PDFs into the Kuzu graph and Qdrant vector store."""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from prefect import flow, task

from graphrag.extraction.entity_extractor import extract
from graphrag.extraction.entity_resolver import resolve_entities
from graphrag.graph.store import (
    get_connection,
    link_chunk_to_entity,
    upsert_chunk,
    upsert_document,
    upsert_entity,
    upsert_relation,
)
from graphrag.ingestion.chunker import chunk_document
from graphrag.ingestion.pdf_loader import load_pdfs_from_folder
from graphrag.vector.store import ensure_collections, get_client
from graphrag.vector.store import upsert_chunk as upsert_chunk_vec

load_dotenv()

logger = logging.getLogger(__name__)


@task
def load_documents(folder_path: str) -> list[dict]:
    """Load all PDFs from a folder and return a list of document dicts."""
    pass


@task
def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk all documents and return a flat list of chunk dicts."""
    pass


@task
def extract_entities(chunks: list[dict]) -> list[dict]:
    """Run entity extraction on each chunk and attach entities to each chunk dict."""
    pass


@task
def store_in_graph(documents: list[dict], chunks: list[dict]) -> None:
    """Insert documents, chunks, entities, and relations into the Kuzu graph."""
    pass


@task
def store_in_vector(chunks: list[dict]) -> None:
    """Upsert chunk embeddings into the Qdrant chunks collection."""
    pass


@flow(name="ingest-pdf-folder")
def ingest_flow(folder_path: str) -> None:
    """End-to-end Prefect flow: load PDFs → chunk → extract → store in graph + vectors."""
    pass
