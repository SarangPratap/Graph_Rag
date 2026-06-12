"""Test extraction on 3 random chunks using the harvesting strategy."""

from __future__ import annotations

import os
import random

from dotenv import load_dotenv
from openai import OpenAI

from graphrag.extraction.entity_extractor import _build_prompt, _parse_response
from graphrag.ingestion.chunker import chunk_documents
from graphrag.ingestion.pdf_loader import load_pdfs_from_folder

load_dotenv()

sarvam = OpenAI(
    api_key=os.environ["SARVAM_API_KEY"],
    base_url="https://api.sarvam.ai/v1",
)


def probe(chunk) -> None:
    """Run extraction on one chunk and print harvested counts."""
    resp = sarvam.chat.completions.create(
        model="sarvam-30b",
        messages=[{"role": "user", "content": _build_prompt(chunk.text)}],
        max_tokens=4096,
        extra_body={"reasoning_effort": "low"},
    )
    msg = resp.choices[0].message
    raw = msg.content or getattr(msg, "reasoning_content", None) or ""
    parsed = _parse_response(raw)
    n_ent = len(parsed["entities"])
    n_rel = len(parsed["relations"])
    first = parsed["entities"][0]["name"] if parsed["entities"] else "—"
    print(f"  chunk {chunk.id[:8]}  entities={n_ent:2d}  relations={n_rel:2d}  first={first!r}")


def main() -> None:
    """Load PDF, pick 3 random chunks, probe each."""
    docs = load_pdfs_from_folder("./sample_pdfs")
    chunks = chunk_documents(docs)
    print(f"\nTotal chunks: {len(chunks)}. Sampling 3 at random.\n")
    for chunk in random.sample(chunks, 3):
        probe(chunk)


if __name__ == "__main__":
    main()
