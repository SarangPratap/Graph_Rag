"""Extract entities and relations from text chunks using Sarvam AI."""

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

_PROMPT_TEMPLATE = """You are an expert data extraction algorithm. Read the text below and extract every entity and relationship you find.

As you work through the text, for every entity you identify immediately write it as a standalone JSON object on its own line:
{"name": "Entity Name", "type": "PERSON|ORG|CONCEPT|LOCATION|EVENT", "description": "brief description"}

For every relationship between two entities, write:
{"source": "Entity A", "target": "Entity B", "relationship": "how they are connected"}

Do not write a final summary or wrap anything in arrays. Just emit one JSON object per line as you discover each item.

Text:
{text_chunk}"""


def extract(chunk: Chunk) -> ExtractionResult:
    """Extract entities and relations from a single Chunk via Sarvam AI.

    Returns an ExtractionResult with empty lists on any failure.
    """
    try:
        prompt = _build_prompt(chunk.text)
        response = sarvam.chat.completions.create(
            model="sarvam-30b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            extra_body={"reasoning_effort": "low"},
        )
        msg = response.choices[0].message
        raw = msg.content or getattr(msg, "reasoning_content", None) or ""
        parsed = _parse_response(raw)
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


def _build_prompt(text: str) -> str:
    """Return the extraction prompt with chunk text interpolated."""
    return _PROMPT_TEMPLATE.replace("{text_chunk}", text)


_PLACEHOLDER_NAMES = {"Entity Name", "Entity A", "Entity B", "Source Name", "Target Name", "Clean Entity Name"}


def _parse_response(raw: str) -> dict:
    """Harvest flat JSON objects from reasoning_content using the per-item strategy.

    Deduplicates by name/source+target so repeated drafts in reasoning don't inflate counts.
    """
    seen_entities: set[str] = set()
    seen_relations: set[tuple] = set()
    entities: list[dict] = []
    relations: list[dict] = []

    for m in re.finditer(r"\{[^{}]+\}", raw, re.DOTALL):
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            continue

        if {"name", "type", "description"} <= obj.keys():
            name = str(obj["name"]).strip()
            if name in _PLACEHOLDER_NAMES or name in seen_entities:
                continue
            seen_entities.add(name)
            entities.append(obj)

        elif {"source", "target", "relationship"} <= obj.keys():
            src = str(obj["source"]).strip()
            tgt = str(obj["target"]).strip()
            key = (src, tgt)
            if src in _PLACEHOLDER_NAMES or tgt in _PLACEHOLDER_NAMES or key in seen_relations:
                continue
            seen_relations.add(key)
            relations.append(obj)

    return {"entities": entities, "relations": relations}


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
