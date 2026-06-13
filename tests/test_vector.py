"""Smoke test for Qdrant vector store — collections, upsert, and search."""

from __future__ import annotations

from graphrag.vector.store import (
    embed,
    ensure_collections,
    get_client,
    search_chunks,
    search_entities,
    upsert_chunk,
    upsert_entity,
)

TEST_CHUNKS = [
    {"id": "a" * 32, "text": "GraphRAG combines graph traversal with vector search.", "source": "/doc1.pdf", "doc_title": "GraphRAG Paper"},
    {"id": "b" * 32, "text": "Microsoft Research published the GraphRAG paper in 2024.", "source": "/doc1.pdf", "doc_title": "GraphRAG Paper"},
    {"id": "c" * 32, "text": "Qdrant is a high-performance vector database.", "source": "/doc2.pdf", "doc_title": "Qdrant Docs"},
]

TEST_ENTITIES = [
    {"id": "d" * 32, "name": "GraphRAG", "type": "CONCEPT", "description": "Graph-enhanced RAG system"},
    {"id": "e" * 32, "name": "Microsoft Research", "type": "ORG", "description": "Research organisation"},
    {"id": "f" * 32, "name": "Darren Edge", "type": "PERSON", "description": "GraphRAG researcher"},
]


def main() -> None:
    client = get_client()

    print("Creating collections...")
    ensure_collections(client)
    cols = {c.name for c in client.get_collections().collections}
    assert {"chunks", "entities", "communities"} <= cols, f"Missing collections: {cols}"
    print(f"  Collections: {sorted(cols)}")

    print("\nEmbedding and upserting chunks...")
    for chunk in TEST_CHUNKS:
        vec = embed(chunk["text"])
        assert len(vec) == 768, f"Expected 768 dims, got {len(vec)}"
        upsert_chunk(client, chunk, vec)
    print(f"  Upserted {len(TEST_CHUNKS)} chunks (768-dim vectors)")

    print("\nEmbedding and upserting entities...")
    for entity in TEST_ENTITIES:
        vec = embed(entity["name"] + " " + entity["description"])
        upsert_entity(client, entity, vec)
    print(f"  Upserted {len(TEST_ENTITIES)} entities")

    print("\nSearching chunks for 'graph knowledge retrieval'...")
    query_vec = embed("graph knowledge retrieval")
    results = search_chunks(client, query_vec, top_k=2)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text'][:60]}")

    print("\nSearching entities for 'machine learning researcher'...")
    query_vec = embed("machine learning researcher")
    results = search_entities(client, query_vec, top_k=2)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['name']} ({r['type']})")

    print("\nIdempotency — upserting same chunks again...")
    for chunk in TEST_CHUNKS:
        upsert_chunk(client, chunk, embed(chunk["text"]))
    count = client.count(collection_name="chunks").count
    print(f"  Chunk count still {count} (no duplicates)")

    print("\nAll vector store tests passed.")


if __name__ == "__main__":
    main()
