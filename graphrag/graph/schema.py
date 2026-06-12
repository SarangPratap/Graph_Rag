"""Kuzu node and edge table definitions for the GraphRAG knowledge graph."""

from __future__ import annotations

import logging

import kuzu

logger = logging.getLogger(__name__)

_NODE_TABLES = [
    """CREATE NODE TABLE IF NOT EXISTS Document(
        id STRING,
        title STRING,
        source STRING,
        checksum STRING,
        ingested_at STRING,
        PRIMARY KEY (id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Chunk(
        id STRING,
        text STRING,
        source STRING,
        doc_title STRING,
        PRIMARY KEY (id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Entity(
        id STRING,
        name STRING,
        type STRING,
        description STRING,
        PRIMARY KEY (id)
    )""",
]

_EDGE_TABLES = [
    "CREATE REL TABLE IF NOT EXISTS CONTAINS(FROM Document TO Chunk)",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Chunk TO Entity)",
    """CREATE REL TABLE IF NOT EXISTS RELATES_TO(
        FROM Entity TO Entity,
        type STRING,
        weight DOUBLE
    )""",
]


def init_schema(conn: kuzu.Connection) -> None:
    """Create all node and edge tables if they do not already exist."""
    _create_node_tables(conn)
    _create_edge_tables(conn)
    logger.info("Kuzu schema initialised")


def _create_node_tables(conn: kuzu.Connection) -> None:
    """Create Document, Chunk, and Entity node tables."""
    for ddl in _NODE_TABLES:
        conn.execute(ddl)


def _create_edge_tables(conn: kuzu.Connection) -> None:
    """Create CONTAINS, MENTIONS, and RELATES_TO edge tables."""
    for ddl in _EDGE_TABLES:
        conn.execute(ddl)
