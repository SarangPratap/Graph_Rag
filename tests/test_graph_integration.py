"""Integration test: extract entities from a real chunk and write them into Kuzu."""

from __future__ import annotations

import os
import shutil

from graphrag.extraction.entity_extractor import extract
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
from graphrag.models import Chunk

TEST_DB = "./test_integration.db"


def cleanup():
    if os.path.isdir(TEST_DB):
        shutil.rmtree(TEST_DB)
    elif os.path.isfile(TEST_DB):
        os.remove(TEST_DB)


def main() -> None:
    cleanup()
    os.environ["KUZU_DB_PATH"] = TEST_DB

    import graphrag.graph.store as store_mod
    store_mod._db = None

    conn = get_connection()

    # A realistic chunk
    chunk = Chunk(
        text=(
            "GraphRAG combines graph-based knowledge representation with retrieval-augmented generation. "
            "Microsoft Research published the original GraphRAG paper in 2024. "
            "Darren Edge and Ha Trinh led the research team at Microsoft. "
            "The technique significantly improves multi-hop question answering compared to naive RAG."
        ),
        source="/papers/graphrag.pdf",
        doc_title="GraphRAG Paper",
        doc_id="doc_graphrag",
        position=0,
    )

    # Step 1: upsert the parent document
    upsert_document(conn, {
        "id": "doc_graphrag",
        "title": "GraphRAG Paper",
        "source": "/papers/graphrag.pdf",
        "checksum": "doc_graphrag",
        "ingested_at": "2026-06-12",
    })

    # Step 2: extract entities via Sarvam
    print("Running entity extraction (2 API calls)...")
    result = extract(chunk)
    print(f"  -> {len(result.entities)} entities, {len(result.relations)} relations\n")

    # Step 3: write chunk into graph
    upsert_chunk(conn, {
        "id": chunk.id,
        "text": chunk.text,
        "source": chunk.source,
        "doc_title": chunk.doc_title,
    }, "doc_graphrag")

    # Step 4: write entities and link to chunk
    for entity in result.entities:
        upsert_entity(conn, {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "description": entity.description,
        })
        link_chunk_to_entity(conn, chunk.id, entity.id)

    # Step 5: write relations
    for rel in result.relations:
        upsert_relation(conn, {
            "source_id": rel.source_id,
            "target_id": rel.target_id,
            "type": rel.type,
            "weight": rel.weight,
        })

    # Step 6: query back
    print("=== Entities stored in Kuzu ===")
    entities_in_graph = get_entities_for_chunk(conn, chunk.id)
    for e in entities_in_graph:
        print(f"  [{e['type']:8s}] {e['name']}")

    print(f"\nTotal: {len(entities_in_graph)} entities linked to chunk\n")

    # Step 7: pick a PERSON entity and expand their neighbourhood
    persons = [e for e in entities_in_graph if e["type"] == "PERSON"]
    if persons:
        anchor = persons[0]
        neighbors = expand_entity_neighbors(conn, anchor["id"], hops=2)
        print(f"=== 2-hop graph from '{anchor['name']}' ===")
        for n in neighbors:
            print(f"  [{n['type']:8s}] {n['name']}")

    cleanup()
    print("\nIntegration test passed.")


if __name__ == "__main__":
    main()
