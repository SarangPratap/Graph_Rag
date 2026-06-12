"""Smoke test for Kuzu graph schema and store."""

from __future__ import annotations

import os
import shutil

from graphrag.graph.store import (
    expand_entity_neighbors,
    get_connection,
    get_entities_for_chunk,
    link_chunk_to_entity,
    upsert_chunk,
    upsert_document,
    upsert_entity,
    upsert_relation,
)

TEST_DB = "./test_graphrag.db"


def main() -> None:
    # Clean slate
    if os.path.isdir(TEST_DB):
        shutil.rmtree(TEST_DB)
    elif os.path.isfile(TEST_DB):
        os.remove(TEST_DB)
    os.environ["KUZU_DB_PATH"] = TEST_DB

    # Reset cached DB instance so get_connection uses the test path
    import graphrag.graph.store as store_mod
    store_mod._db = None

    conn = get_connection()
    print("Schema initialised.")

    upsert_document(conn, {
        "id": "doc1", "title": "GraphRAG Paper",
        "source": "/papers/graphrag.pdf", "checksum": "doc1", "ingested_at": "2026-06-12",
    })

    upsert_chunk(conn, {"id": "chunk1", "text": "GraphRAG by Microsoft Research.", "source": "/papers/graphrag.pdf", "doc_title": "GraphRAG Paper"}, "doc1")
    upsert_chunk(conn, {"id": "chunk2", "text": "Darren Edge and Ha Trinh led the team.", "source": "/papers/graphrag.pdf", "doc_title": "GraphRAG Paper"}, "doc1")

    upsert_entity(conn, {"id": "e1", "name": "GraphRAG", "type": "CONCEPT", "description": "Graph RAG system"})
    upsert_entity(conn, {"id": "e2", "name": "Microsoft Research", "type": "ORG", "description": "Research org"})
    upsert_entity(conn, {"id": "e3", "name": "Darren Edge", "type": "PERSON", "description": "Researcher"})

    link_chunk_to_entity(conn, "chunk1", "e1")
    link_chunk_to_entity(conn, "chunk1", "e2")
    link_chunk_to_entity(conn, "chunk2", "e3")

    upsert_relation(conn, {"source_id": "e3", "target_id": "e2", "type": "WORKS_AT", "weight": 0.9})
    upsert_relation(conn, {"source_id": "e2", "target_id": "e1", "type": "DEVELOPED", "weight": 1.0})

    entities = get_entities_for_chunk(conn, "chunk1")
    print(f"Entities in chunk1 : {[e['name'] for e in entities]}")

    neighbors = expand_entity_neighbors(conn, "e3", hops=2)
    print(f"2-hop from Darren  : {[n['name'] for n in neighbors]}")

    # Idempotency check — run again, counts must stay the same
    upsert_document(conn, {"id": "doc1", "title": "GraphRAG Paper", "source": "/papers/graphrag.pdf", "checksum": "doc1", "ingested_at": "2026-06-12"})
    upsert_entity(conn, {"id": "e1", "name": "GraphRAG", "type": "CONCEPT", "description": "Graph RAG system"})
    entities2 = get_entities_for_chunk(conn, "chunk1")
    assert len(entities2) == len(entities), "MERGE is not idempotent!"
    print("Idempotency check  : passed")

    if os.path.isdir(TEST_DB):
        shutil.rmtree(TEST_DB)
    elif os.path.isfile(TEST_DB):
        os.remove(TEST_DB)
    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
