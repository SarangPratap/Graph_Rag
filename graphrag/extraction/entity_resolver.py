"""Resolve duplicate entities via Qdrant cosine similarity before graph insertion."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.92


def resolve_entities(
    entities: list[dict[str, Any]],
    qdrant_client: QdrantClient,
) -> list[dict[str, Any]]:
    """Deduplicate entities against the Qdrant entities collection and return canonical list."""
    pass


def entity_id(name: str) -> str:
    """Return MD5 hash of lowercased entity name as the canonical entity ID."""
    pass


def _embed_name(name: str) -> list[float]:
    """Embed an entity name string for similarity lookup."""
    pass
