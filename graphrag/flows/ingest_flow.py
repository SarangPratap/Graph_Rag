"""Prefect flow: ingest a folder of PDFs into the Kuzu graph and Qdrant vector store."""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from prefect import flow, task

from graphrag.extraction.entity_extractor import extract_batch_parallel
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
from graphrag.models import Chunk, Document, ExtractionResult
from graphrag.vector.store import (
    embed,
    ensure_collections,
    get_client,
    upsert_chunk as upsert_chunk_vec,
    upsert_entity as upsert_entity_vec,
)

load_dotenv()

logger = logging.getLogger(__name__)


@task
def load_documents(folder_path: str) -> list[Document]:
    """Load all PDFs from a folder and return a list of Document instances."""
    docs = load_pdfs_from_folder(folder_path)
    print(f"Loaded {len(docs)} document(s)")
    return docs


@task
def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Chunk all documents and return a flat list of Chunk instances."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    print(f"Produced {len(all_chunks)} chunk(s)")
    return all_chunks


@task
def extract_entities(chunks: list[Chunk]) -> list[ExtractionResult]:
    """Run entity extraction on all chunks in parallel (5 workers)."""
    results = extract_batch_parallel(chunks, max_workers=5)
    total_ents = sum(len(r.entities) for r in results)
    total_rels = sum(len(r.relations) for r in results)
    print(f"Extracted {total_ents} entities and {total_rels} relations")
    return results


@task
def store_in_graph(
    documents: list[Document],
    chunks: list[Chunk],
    results: list[ExtractionResult],
) -> None:
    """Insert documents, chunks, entities, and relations into the Kuzu graph."""
    conn = get_connection()
    qdrant = get_client()

    chunk_map = {c.id: c for c in chunks}

    for doc in documents:
        upsert_document(conn, doc.model_dump(mode="json"))

    result_map = {r.chunk_id: r for r in results}

    for chunk in chunks:
        upsert_chunk(conn, chunk.model_dump(mode="json"), chunk.doc_id)

        result = result_map.get(chunk.id)
        if not result:
            continue

        entity_dicts = [
            {"id": e.id, "name": e.name, "type": e.type, "description": e.description}
            for e in result.entities
        ]
        resolved = resolve_entities(entity_dicts, qdrant)

        id_map = {orig["id"]: res["id"] for orig, res in zip(entity_dicts, resolved)}

        for ent in resolved:
            upsert_entity(conn, ent)
            link_chunk_to_entity(conn, chunk.id, ent["id"])

        for rel in result.relations:
            src_id = id_map.get(rel.source_id, rel.source_id)
            tgt_id = id_map.get(rel.target_id, rel.target_id)
            upsert_relation(conn, {
                "source_id": src_id,
                "target_id": tgt_id,
                "type": rel.type,
                "weight": rel.weight,
            })

    print("Graph store complete")


@task
def store_in_vector(
    chunks: list[Chunk],
    results: list[ExtractionResult],
) -> None:
    """Upsert chunk and entity embeddings into Qdrant."""
    client = get_client()
    ensure_collections(client)

    result_map = {r.chunk_id: r for r in results}

    for chunk in chunks:
        vec = embed(chunk.text)
        upsert_chunk_vec(client, chunk.model_dump(mode="json"), vec)

    seen_entity_ids: set[str] = set()
    for result in results:
        for ent in result.entities:
            if ent.id in seen_entity_ids:
                continue
            seen_entity_ids.add(ent.id)
            vec = embed(ent.name + " " + ent.description)
            upsert_entity_vec(
                client,
                {"id": ent.id, "name": ent.name, "type": ent.type, "description": ent.description},
                vec,
            )

    print(f"Vector store complete — {len(chunks)} chunks, {len(seen_entity_ids)} entities")


@flow(name="ingest-pdf-folder")
def ingest_flow(folder_path: str) -> None:
    """End-to-end flow: load PDFs → chunk → extract → store in graph + vectors."""
    docs = load_documents(folder_path)
    if not docs:
        print("No documents found, exiting.")
        return

    chunks = chunk_documents(docs)
    results = extract_entities(chunks)

    store_in_graph(docs, chunks, results)
    store_in_vector(chunks, results)

    print(f"\nIngestion complete: {len(docs)} docs, {len(chunks)} chunks")
