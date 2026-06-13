"""Qdrant vector store: collection management, upsert, and similarity search."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

logger = logging.getLogger(__name__)

VECTOR_SIZE = 768
COLLECTIONS = ["chunks", "entities", "communities"]

# Lazy-loaded embedding model — initialised on first call to embed()
_model = None


def _get_model():
    """Return a shared SentenceTransformer instance, creating it on first call."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        logger.info("Embedding model loaded: all-mpnet-base-v2")
    return _model


def embed(text: str) -> list[float]:
    """Embed a single string and return a 768-dim float vector."""
    return _get_model().encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings and return a list of 768-dim float vectors."""
    return _get_model().encode(texts, normalize_embeddings=True).tolist()


def _to_uuid(hex_id: str) -> str:
    """Convert an MD5 hex string (32 chars) to a UUID string for Qdrant point IDs."""
    return str(uuid.UUID(hex_id))


def get_client() -> QdrantClient:
    """Return a Qdrant client.

    Prefers QDRANT_URL + optional QDRANT_API_KEY (Qdrant Cloud).
    Falls back to QDRANT_HOST / QDRANT_PORT for local Docker.
    """
    url = os.environ.get("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", 6333))
    return QdrantClient(host=host, port=port)


def ensure_collections(client: QdrantClient) -> None:
    """Create all required Qdrant collections if they do not already exist."""
    existing = {c.name for c in client.get_collections().collections}
    for name in COLLECTIONS:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", name)
        else:
            logger.debug("Collection already exists: %s", name)


def upsert_chunk(client: QdrantClient, chunk: dict[str, Any], vector: list[float]) -> None:
    """Upsert a chunk embedding and payload into the chunks collection."""
    client.upsert(
        collection_name="chunks",
        points=[PointStruct(
            id=_to_uuid(chunk["id"]),
            vector=vector,
            payload={
                "id": chunk["id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "doc_title": chunk["doc_title"],
            },
        )],
    )


def upsert_entity(client: QdrantClient, entity: dict[str, Any], vector: list[float]) -> None:
    """Upsert an entity embedding and payload into the entities collection."""
    client.upsert(
        collection_name="entities",
        points=[PointStruct(
            id=_to_uuid(entity["id"]),
            vector=vector,
            payload={
                "id": entity["id"],
                "name": entity["name"],
                "type": entity["type"],
                "description": entity.get("description", ""),
            },
        )],
    )


def search_chunks(
    client: QdrantClient, query_vector: list[float], top_k: int = 5
) -> list[dict[str, Any]]:
    """Return the top-k most similar chunks for a query vector."""
    response = client.query_points(
        collection_name="chunks",
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [{"score": h.score, **h.payload} for h in response.points]


def search_entities(
    client: QdrantClient, query_vector: list[float], top_k: int = 5
) -> list[dict[str, Any]]:
    """Return the top-k most similar entities for a query vector."""
    response = client.query_points(
        collection_name="entities",
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [{"score": h.score, **h.payload} for h in response.points]
