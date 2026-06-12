"""Extract entities and relations from text chunks using Sarvam AI (two-turn strategy)."""

from __future__ import annotations

import json
import logging
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from graphrag.models import Chunk, Entity, ExtractionResult, Relation

load_dotenv()

logger = logging.getLogger(__name__)

sarvam = OpenAI(
    api_key=os.environ["SARVAM_API_KEY"],
    base_url="https://api.sarvam.ai/v1",
)

_VALID_TYPES = {"PERSON", "ORG", "CONCEPT", "LOCATION", "EVENT"}

# Turn 1 — free reasoning, no format pressure
_EXTRACTION_PROMPT = """You are an expert knowledge extraction system. Carefully read the text below and identify:
- Every named entity (people, organizations, concepts, locations, events)
- Every meaningful relationship between those entities

Think through the text thoroughly. There is no output format required — just reason carefully.

Text:
{text_chunk}"""

# Turn 2 — model sees its own reasoning, just formats the final answer
_FORMAT_PROMPT = """Based on your analysis above, output ONLY the following JSON object with no explanation, no markdown, no extra text:

{
  "entities": [
    {"name": "Entity Name", "type": "PERSON|ORG|CONCEPT|LOCATION|EVENT", "description": "brief description"}
  ],
  "relations": [
    {"source": "Entity A", "target": "Entity B", "relationship": "how they connect"}
  ]
}

Use only these entity types: PERSON, ORG, CONCEPT, LOCATION, EVENT."""


def extract(chunk: Chunk) -> ExtractionResult:
    """Extract entities and relations from a single Chunk using a two-turn Sarvam conversation.

    Turn 1 lets the model reason freely. Turn 2 asks it to format its findings as
    clean JSON — short response, no token pressure, reliable output.
    Returns an ExtractionResult with empty lists on any failure.
    """
    try:
        extraction_prompt = _EXTRACTION_PROMPT.replace("{text_chunk}", chunk.text)

        # Turn 1: reason freely about the text
        resp1 = sarvam.chat.completions.create(
            model="sarvam-30b",
            messages=[{"role": "user", "content": extraction_prompt}],
            max_tokens=4096,
            extra_body={"reasoning_effort": "low"},
        )
        msg1 = resp1.choices[0].message
        thinking = msg1.content or getattr(msg1, "reasoning_content", None) or ""

        # Turn 2: convert reasoning into final JSON
        resp2 = sarvam.chat.completions.create(
            model="sarvam-30b",
            messages=[
                {"role": "user",      "content": extraction_prompt},
                {"role": "assistant", "content": thinking},
                {"role": "user",      "content": _FORMAT_PROMPT},
            ],
            max_tokens=4096,
            extra_body={"reasoning_effort": "low"},
        )
        msg2 = resp2.choices[0].message
        raw = msg2.content or getattr(msg2, "reasoning_content", None) or ""

        parsed = _parse_json(raw)
        return _build_result(chunk, parsed)

    except Exception as exc:
        logger.error("Extraction failed for chunk %s: %s", chunk.id, exc)
        return ExtractionResult(chunk_id=chunk.id)


def extract_batch(chunks: list[Chunk]) -> list[ExtractionResult]:
    """Extract entities and relations from every Chunk in the list."""
    total = len(chunks)
    results: list[ExtractionResult] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Extracting chunk {i}/{total}...")
        results.append(extract(chunk))
    return results


def _parse_json(raw: str) -> dict:
    """Strip fences and <think> tags, then parse the first balanced JSON object."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    text = text.replace("```json", "").replace("```", "").strip()

    # Find the outermost { ... } that contains "entities"
    start = text.find('{"entities"')
    if start == -1:
        start = text.find('{')
    if start == -1:
        logger.warning("No JSON object found in response")
        return {"entities": [], "relations": []}

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as exc:
                    logger.warning("JSON parse failed: %s", exc)
                    return {"entities": [], "relations": []}

    logger.warning("Unbalanced JSON in response")
    return {"entities": [], "relations": []}


def _build_result(chunk: Chunk, parsed: dict) -> ExtractionResult:
    """Convert a parsed JSON dict into an ExtractionResult."""
    entities: list[Entity] = []
    name_to_id: dict[str, str] = {}

    for raw_ent in parsed.get("entities", []):
        try:
            ent_type = str(raw_ent.get("type", "CONCEPT")).upper()
            if ent_type not in _VALID_TYPES:
                ent_type = "CONCEPT"
            entity = Entity(
                name=raw_ent["name"],
                type=ent_type,
                description=raw_ent.get("description", ""),
                chunk_id=chunk.id,
            )
            entities.append(entity)
            name_to_id[raw_ent["name"]] = entity.id
        except Exception as exc:
            logger.warning("Skipping malformed entity %s: %s", raw_ent, exc)

    relations: list[Relation] = []
    for raw_rel in parsed.get("relations", []):
        try:
            src_name = raw_rel.get("source") or raw_rel.get("source_name", "")
            tgt_name = raw_rel.get("target") or raw_rel.get("target_name", "")
            src_id = name_to_id.get(src_name)
            tgt_id = name_to_id.get(tgt_name)
            if not src_id or not tgt_id:
                continue
            rel_type = raw_rel.get("relationship") or raw_rel.get("type", "RELATES_TO")
            weight = max(0.0, min(1.0, float(raw_rel.get("weight", 0.5))))
            relations.append(Relation(
                source_id=src_id,
                target_id=tgt_id,
                type=rel_type,
                weight=weight,
            ))
        except Exception as exc:
            logger.warning("Skipping malformed relation %s: %s", raw_rel, exc)

    return ExtractionResult(chunk_id=chunk.id, entities=entities, relations=relations)
