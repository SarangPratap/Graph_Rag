"""Unit tests for the query planner using real Qdrant + Kuzu connections."""

from __future__ import annotations

import os
import shutil

import pytest

from graphrag.graph.store import (
    get_connection,
    link_chunk_to_entity,
    upsert_chunk,
    upsert_document,
    upsert_entity,
    upsert_relation,
)
from graphrag.models import Chunk
from graphrag.query.planner import answer_question
from graphrag.vector.store import embed, ensure_collections, get_client, upsert_chunk as upsert_chunk_vec

TEST_DB = "./test_query.db"

_CHUNK_TEXT = (
    "GraphRAG combines graph-based knowledge representation with retrieval-augmented generation. "
    "Microsoft Research published the original GraphRAG paper in 2024. "
    "Darren Edge and Ha Trinh led the research team. "
    "The technique improves multi-hop question answering compared to naive RAG."
)

_DOC_ID = "doc_test_query"


@pytest.fixture(scope="module")
def stores():
    """Set up a fresh Kuzu DB and populate Qdrant with one chunk + two entities."""
    # --- Kuzu setup ---
    if os.path.isdir(TEST_DB):
        shutil.rmtree(TEST_DB)
    os.environ["KUZU_DB_PATH"] = TEST_DB

    import graphrag.graph.store as store_mod
    store_mod._db = None

    conn = get_connection()

    chunk = Chunk(
        text=_CHUNK_TEXT,
        source="/papers/graphrag.pdf",
        doc_title="GraphRAG Paper",
        doc_id=_DOC_ID,
        position=0,
    )

    upsert_document(conn, {
        "id": _DOC_ID,
        "title": "GraphRAG Paper",
        "source": "/papers/graphrag.pdf",
        "checksum": _DOC_ID,
        "ingested_at": "2026-06-13",
    })
    upsert_chunk(conn, chunk.model_dump(mode="json"), _DOC_ID)

    entity_a = {"id": "e_graphrag", "name": "GraphRAG", "type": "CONCEPT", "description": "Graph-based RAG technique"}
    entity_b = {"id": "e_msft",    "name": "Microsoft Research", "type": "ORG",     "description": "Research division of Microsoft"}

    for ent in (entity_a, entity_b):
        upsert_entity(conn, ent)
        link_chunk_to_entity(conn, chunk.id, ent["id"])

    upsert_relation(conn, {"source_id": "e_graphrag", "target_id": "e_msft", "type": "PUBLISHED_BY", "weight": 0.9})

    # --- Qdrant setup ---
    qdrant = get_client()
    ensure_collections(qdrant)
    vec = embed(_CHUNK_TEXT)
    upsert_chunk_vec(qdrant, chunk.model_dump(mode="json"), vec)

    yield qdrant, conn, chunk

    # --- teardown ---
    if os.path.isdir(TEST_DB):
        shutil.rmtree(TEST_DB)


def test_answer_question_returns_string(stores) -> None:
    """answer_question must return a non-empty string."""
    qdrant, conn, _ = stores
    result = answer_question("What is GraphRAG?", qdrant, conn)
    assert isinstance(result, str)
    assert len(result.strip()) > 0


def test_answer_question_uses_graph_context(stores) -> None:
    """The answer should mention entities that were linked in the graph."""
    qdrant, conn, _ = stores
    result = answer_question("Who published GraphRAG?", qdrant, conn)
    # At minimum the answer should reference Microsoft or the paper
    lower = result.lower()
    assert "microsoft" in lower or "graphrag" in lower or "research" in lower
