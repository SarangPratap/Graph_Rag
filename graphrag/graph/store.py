"""Kuzu graph store: CRUD operations and Cypher query helpers."""

from __future__ import annotations

import logging
import os
from typing import Any

import kuzu
from dotenv import load_dotenv

from graphrag.graph.schema import init_schema

load_dotenv()

logger = logging.getLogger(__name__)


def get_connection() -> kuzu.Connection:
    """Open and return a Kuzu database connection using KUZU_DB_PATH from env."""
    pass


def upsert_document(conn: kuzu.Connection, doc: dict[str, Any]) -> None:
    """Merge a Document node into the graph (MERGE, not CREATE)."""
    pass


def upsert_chunk(conn: kuzu.Connection, chunk: dict[str, Any], doc_id: str) -> None:
    """Merge a Chunk node and CONTAINS edge into the graph."""
    pass


def upsert_entity(conn: kuzu.Connection, entity: dict[str, Any]) -> None:
    """Merge an Entity node into the graph."""
    pass


def upsert_relation(conn: kuzu.Connection, relation: dict[str, Any]) -> None:
    """Merge a RELATES_TO edge between two Entity nodes."""
    pass


def link_chunk_to_entity(conn: kuzu.Connection, chunk_id: str, entity_id: str) -> None:
    """Create a MENTIONS edge from a Chunk node to an Entity node."""
    pass


def get_entities_for_chunk(conn: kuzu.Connection, chunk_id: str) -> list[dict[str, Any]]:
    """Return all entities mentioned by a given chunk."""
    pass


def expand_entity_neighbors(
    conn: kuzu.Connection, entity_id: str, hops: int = 2
) -> list[dict[str, Any]]:
    """Return all entities reachable within N hops via RELATES_TO edges."""
    pass
