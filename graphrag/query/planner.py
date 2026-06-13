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
from graphrag.vector.store import embed, search_chunks

load_dotenv()

logger = logging.getLogger(__name__)

sarvam = OpenAI(
    api_key=os.environ["SARVAM_API_KEY"],
    base_url="https://api.sarvam.ai/v1",
)

_ANSWER_PROMPT = """You are a helpful research assistant with access to a knowledge graph and document database. Use the context below to answer the question accurately and concisely. If the context does not contain enough information to answer, say so.

Context:
{context}

Question: {question}

Answer:"""


def answer_question(
    question: str,
    qdrant_client: QdrantClient,
    kuzu_conn: kuzu.Connection,
) -> str:
    """Answer a question by fusing vector search results with graph traversal context."""
    # Step 1: embed question → find top relevant chunks
    q_vec = _embed_question(question)
    chunks = search_chunks(qdrant_client, q_vec, top_k=5)
    if not chunks:
        logger.warning("No chunks found for question: %s", question)

    # Steps 2 & 3: collect anchor entities from chunks, expand 2-hop neighbors
    seen_ids: set[str] = set()
    subgraph: list[dict[str, Any]] = []

    for chunk in chunks:
        chunk_id = chunk.get("id", "")
        if not chunk_id:
            continue
        for ent in get_entities_for_chunk(kuzu_conn, chunk_id):
            if ent["id"] not in seen_ids:
                seen_ids.add(ent["id"])
                subgraph.append(ent)
            for neighbor in expand_entity_neighbors(kuzu_conn, ent["id"], hops=2):
                if neighbor["id"] not in seen_ids:
                    seen_ids.add(neighbor["id"])
                    subgraph.append(neighbor)

    # Step 4: pull community summaries if the collection is populated
    summaries: list[dict[str, Any]] = []
    try:
        response = qdrant_client.query_points(
            collection_name="communities",
            query=q_vec,
            limit=3,
            with_payload=True,
        )
        summaries = [{"score": h.score, **h.payload} for h in response.points]
    except Exception as exc:
        logger.debug("Community summaries unavailable: %s", exc)

    # Steps 5 & 6: build context string and call Sarvam
    context = _build_context(chunks, subgraph, summaries)
    return _call_sarvam(context, question)


def _embed_question(question: str) -> list[float]:
    """Embed a question string for Qdrant similarity search."""
    return embed(question)


def _build_context(
    chunks: list[dict[str, Any]],
    subgraph: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> str:
    """Assemble a context string from chunks, subgraph entities, and community summaries."""
    parts: list[str] = []

    if chunks:
        parts.append("## Relevant Passages")
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("doc_title", "Unknown")
            text = chunk.get("text", "").strip()
            parts.append(f"[{i}] ({title})\n{text}")

    if subgraph:
        parts.append("## Related Entities from Knowledge Graph")
        # cap at 20 entities to avoid blowing the context window
        for ent in subgraph[:20]:
            line = f"- {ent.get('name', '')} ({ent.get('type', '')}): {ent.get('description', '')}"
            parts.append(line)

    if summaries:
        parts.append("## Community Summaries")
        for s in summaries:
            text = s.get("summary", s.get("text", "")).strip()
            if text:
                parts.append(text)

    return "\n\n".join(parts)


def _call_sarvam(context: str, question: str) -> str:
    """Send context + question to Sarvam and return the answer string."""
    prompt = _ANSWER_PROMPT.replace("{context}", context).replace("{question}", question)
    try:
        resp = sarvam.chat.completions.create(
            model="sarvam-30b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            extra_body={"reasoning_effort": "low"},
        )
        msg = resp.choices[0].message
        return msg.content or getattr(msg, "reasoning_content", None) or ""
    except Exception as exc:
        logger.error("Sarvam answer call failed: %s", exc)
        return f"Error generating answer: {exc}"
