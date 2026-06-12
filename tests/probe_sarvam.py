"""Minimal Sarvam probe — tests the harvest strategy with a hardcoded text snippet."""

from __future__ import annotations

from graphrag.extraction.entity_extractor import _build_prompt, _parse_response
from graphrag.models import Chunk
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sarvam = OpenAI(
    api_key=os.environ["SARVAM_API_KEY"],
    base_url="https://api.sarvam.ai/v1",
)

TEXT = (
    "GraphRAG combines graph-based knowledge representation with retrieval-augmented generation. "
    "Microsoft Research published the original GraphRAG paper in 2024. The system uses Neo4j as "
    "its graph database and OpenAI embeddings for vector search. Darren Edge and Ha Trinh led the "
    "research team at Microsoft. The technique significantly improves multi-hop question answering "
    "compared to naive RAG."
)

chunk = Chunk(text=TEXT, source="probe", doc_title="probe", doc_id="probe", position=0)

print("=== SENDING REQUEST TO SARVAM ===\n")

response = sarvam.chat.completions.create(
    model="sarvam-30b",
    messages=[{"role": "user", "content": _build_prompt(chunk.text)}],
    max_tokens=4096,
    extra_body={"reasoning_effort": "low"},
)

msg = response.choices[0].message
raw = msg.content or getattr(msg, "reasoning_content", None) or ""

print(f"msg.content  : {repr(msg.content)}")
print(f"raw length   : {len(raw)} chars\n")

result = _parse_response(raw)
print(f"Entities  ({len(result['entities'])}):")
for e in result["entities"]:
    print(f"  {e['name']!r:30s} [{e['type']}]  {e['description'][:60]}")

print(f"\nRelations ({len(result['relations'])}):")
for r in result["relations"]:
    print(f"  {r['source']!r} -> {r['target']!r}  ({r['relationship']})")
