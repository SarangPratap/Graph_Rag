"""Debug a specific chunk — shows raw reasoning and harvested entities/relations."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from graphrag.extraction.entity_extractor import _build_prompt, _parse_response
from graphrag.ingestion.chunker import chunk_documents
from graphrag.ingestion.pdf_loader import load_pdfs_from_folder

load_dotenv()

TARGET_ID = "e7907ddf"  # chunk that previously returned 0 entities

sarvam = OpenAI(
    api_key=os.environ["SARVAM_API_KEY"],
    base_url="https://api.sarvam.ai/v1",
)


def main() -> None:
    docs = load_pdfs_from_folder("./sample_pdfs")
    chunks = chunk_documents(docs)

    chunk = next((c for c in chunks if c.id.startswith(TARGET_ID)), None)
    if chunk is None:
        print(f"Chunk {TARGET_ID} not found — IDs shifted. Using first chunk.")
        chunk = chunks[0]

    print(f"\n=== CHUNK TEXT (first 400 chars) ===\n{chunk.text[:400]}\n")

    resp = sarvam.chat.completions.create(
        model="sarvam-30b",
        messages=[{"role": "user", "content": _build_prompt(chunk.text)}],
        max_tokens=4096,
        extra_body={"reasoning_effort": "low"},
    )
    msg = resp.choices[0].message
    raw = msg.content or getattr(msg, "reasoning_content", None) or ""

    print(f"=== RAW RESPONSE (last 600 chars) ===\n{raw[-600:]}\n")

    parsed = _parse_response(raw)
    print(f"\n=== RESULT ===")
    print(f"  entities : {len(parsed['entities'])}")
    print(f"  relations: {len(parsed['relations'])}")
    if parsed["entities"]:
        print(f"  first entity: {parsed['entities'][0]}")


if __name__ == "__main__":
    main()
