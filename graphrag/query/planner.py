"""Query planner: fuses Qdrant vector search and Kuzu graph traversal to answer questions via Sarvam."""

from __future__ import annotations

import logging
import os
from typing import Any

import kuzu
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from graphrag.graph.store import expand_entity_neighbors, get_entities_for_chunk
from graphrag.vector.store import search_chunks

load_dotenv()

logger = logging.getLogger(__name__)

sarvam = OpenAI(
    api_key=os.environ["SARVAM_API_KEY"],
    base_url="https://api.sarvam.ai/v1",
)


def answer_question(
    question: str,
    qdrant_client: QdrantClient,
    kuzu_conn: kuzu.Connection,
) -> str:
    """Answer a question by fusing vector search results with graph traversal context."""
    pass


def _embed_question(question: str) -> list[float]:
    """Embed a question string for Qdrant similarity search."""
    pass


def _build_context(
    chunks: list[dict[str, Any]],
    subgraph: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> str:
    """Assemble a context string from chunks, subgraph entities, and community summaries."""
    pass


def _call_sarvam(context: str, question: str) -> str:
    """Send context + question to Sarvam and return the answer string."""
    pass
