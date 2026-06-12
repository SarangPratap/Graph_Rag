"""Resolve duplicate entities via Qdrant cosine similarity before graph insertion."""

from __future__ import annotations

import hashlib
import logging

from qdrant_client import QdrantClient

from graphrag.vector.store import embed, search_entities

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.92


def resolve_entities(
    entities: list[dict],
    qdrant_client: QdrantClient,
) -> list[dict]:
    """Deduplicate entities against the Qdrant entities collection.

    For each entity, embeds the name and searches Qdrant. If a match above
    the similarity threshold exists, the canonical ID from Qdrant is used
    instead of the new one — preventing duplicate nodes in the graph.
    """
    resolved = []
    for ent in entities:
        try:
            vec = embed(ent["name"] + " " + ent.get("description", ""))
            hits = search_entities(qdrant_client, vec, top_k=1)
            if hits and hits[0]["score"] >= SIMILARITY_THRESHOLD:
                canonical_id = hits[0]["id"]
                if canonical_id != ent["id"]:
                    logger.debug(
                        "Resolved %r -> existing entity %s (score %.3f)",
                        ent["name"], canonical_id, hits[0]["score"],
                    )
                    ent = {**ent, "id": canonical_id}
        except Exception as exc:
            logger.warning("Resolution failed for %r: %s", ent.get("name"), exc)
        resolved.append(ent)
    return resolved


def entity_id(name: str) -> str:
    """Return MD5 hash of lowercased entity name as the canonical entity ID."""
    return hashlib.md5(name.lower().encode()).hexdigest()


def _embed_name(name: str) -> list[float]:
    """Embed an entity name string for similarity lookup."""
    return embed(name)
