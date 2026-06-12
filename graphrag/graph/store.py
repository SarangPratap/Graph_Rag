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

_db: kuzu.Database | None = None


def get_connection() -> kuzu.Connection:
    """Open and return a Kuzu database connection using KUZU_DB_PATH from env."""
    global _db
    db_path = os.environ.get("KUZU_DB_PATH", "./graphrag.db")
    if _db is None:
        _db = kuzu.Database(db_path)
    conn = kuzu.Connection(_db)
    init_schema(conn)
    return conn


def upsert_document(conn: kuzu.Connection, doc: dict[str, Any]) -> None:
    """Merge a Document node into the graph (MERGE, not CREATE)."""
    conn.execute(
        """
        MERGE (d:Document {id: $id})
        SET d.title = $title,
            d.source = $source,
            d.checksum = $checksum,
            d.ingested_at = $ingested_at
        """,
        {
            "id": doc["id"],
            "title": doc["title"],
            "source": doc["source"],
            "checksum": doc.get("checksum", doc["id"]),
            "ingested_at": str(doc.get("ingested_at", "")),
        },
    )


def upsert_chunk(conn: kuzu.Connection, chunk: dict[str, Any], doc_id: str) -> None:
    """Merge a Chunk node and CONTAINS edge into the graph."""
    conn.execute(
        """
        MERGE (c:Chunk {id: $id})
        SET c.text = $text,
            c.source = $source,
            c.doc_title = $doc_title
        """,
        {
            "id": chunk["id"],
            "text": chunk["text"],
            "source": chunk["source"],
            "doc_title": chunk["doc_title"],
        },
    )
    conn.execute(
        """
        MATCH (d:Document {id: $doc_id}), (c:Chunk {id: $chunk_id})
        MERGE (d)-[:CONTAINS]->(c)
        """,
        {"doc_id": doc_id, "chunk_id": chunk["id"]},
    )


def upsert_entity(conn: kuzu.Connection, entity: dict[str, Any]) -> None:
    """Merge an Entity node into the graph."""
    conn.execute(
        """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.type = $type,
            e.description = $description
        """,
        {
            "id": entity["id"],
            "name": entity["name"],
            "type": entity["type"],
            "description": entity.get("description", ""),
        },
    )


def upsert_relation(conn: kuzu.Connection, relation: dict[str, Any]) -> None:
    """Merge a RELATES_TO edge between two Entity nodes."""
    conn.execute(
        """
        MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id})
        MERGE (a)-[r:RELATES_TO]->(b)
        SET r.type = $type,
            r.weight = $weight
        """,
        {
            "source_id": relation["source_id"],
            "target_id": relation["target_id"],
            "type": relation.get("type", "RELATES_TO"),
            "weight": float(relation.get("weight", 0.5)),
        },
    )


def link_chunk_to_entity(conn: kuzu.Connection, chunk_id: str, entity_id: str) -> None:
    """Create a MENTIONS edge from a Chunk node to an Entity node."""
    conn.execute(
        """
        MATCH (c:Chunk {id: $chunk_id}), (e:Entity {id: $entity_id})
        MERGE (c)-[:MENTIONS]->(e)
        """,
        {"chunk_id": chunk_id, "entity_id": entity_id},
    )


def get_entities_for_chunk(conn: kuzu.Connection, chunk_id: str) -> list[dict[str, Any]]:
    """Return all entities mentioned by a given chunk."""
    result = conn.execute(
        """
        MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e:Entity)
        RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description
        """,
        {"chunk_id": chunk_id},
    )
    rows = []
    while result.has_next():
        row = result.get_next()
        rows.append({"id": row[0], "name": row[1], "type": row[2], "description": row[3]})
    return rows


def expand_entity_neighbors(
    conn: kuzu.Connection, entity_id: str, hops: int = 2
) -> list[dict[str, Any]]:
    """Return all entities reachable within N hops via RELATES_TO edges."""
    result = conn.execute(
        f"""
        MATCH (start:Entity {{id: $entity_id}})-[:RELATES_TO*1..{hops}]->(neighbor:Entity)
        RETURN DISTINCT neighbor.id AS id, neighbor.name AS name,
               neighbor.type AS type, neighbor.description AS description
        """,
        {"entity_id": entity_id},
    )
    rows = []
    while result.has_next():
        row = result.get_next()
        rows.append({"id": row[0], "name": row[1], "type": row[2], "description": row[3]})
    return rows
