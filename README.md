# GraphRAG — Graph-Enhanced Retrieval-Augmented Generation

A production-grade knowledge graph + vector search pipeline that answers complex, multi-hop questions over large document collections.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What Problem Does This Solve?

Standard RAG retrieves the top-K most similar text chunks to a question and feeds them to an LLM. This breaks on questions that require connecting information spread across multiple documents — because no single chunk contains the full answer.

GraphRAG solves this by building a **knowledge graph** on top of your documents. When a question arrives, it finds relevant chunks via vector search, then **walks the graph** to pull in related entities and context that naive RAG would miss.

---

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                        INGESTION PIPELINE                       │
 │                                                                 │
 │  PDF / Web  ──► Docling / Crawl4AI  ──► Clean Text             │
 │                                              │                  │
 │                                         HybridChunker           │
 │                                              │                  │
 │                                       ~400-token Chunks         │
 │                                         /         \             │
 │                                  Sarvam AI       Embeddings     │
 │                                  Extraction          │          │
 │                                 /        \           │          │
 │                           Entities    Relations   Qdrant        │
 │                               │           │      (chunks)       │
 │                           Entity        Kuzu                    │
 │                          Resolver    (graph DB)                 │
 │                               └─────────┘                       │
 └─────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────┐
 │                          QUERY PIPELINE                         │
 │                                                                 │
 │  Question  ──► Embed  ──► Qdrant search  ──► top-5 chunks      │
 │                                                    │            │
 │                                          Kuzu: MENTIONS edges   │
 │                                                    │            │
 │                                        Anchor entities found    │
 │                                                    │            │
 │                                    Graph traversal (2 hops)     │
 │                                                    │            │
 │                                        Subgraph + summaries     │
 │                                                    │            │
 │                               chunks + subgraph + summaries     │
 │                                                    │            │
 │                                              Sarvam AI          │
 │                                                    │            │
 │                                               Answer ◄──────────┘
 └─────────────────────────────────────────────────────────────────┘
```

---

## Stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | [Sarvam AI](https://sarvam.ai) `sarvam-30b` | OpenAI-compatible reasoning model |
| Graph DB | [Kuzu](https://kuzudb.com) | Embedded, no server needed |
| Vector DB | [Qdrant](https://qdrant.tech) | Runs via Docker |
| PDF ingestion | [Docling](https://github.com/DS4SD/docling) | Layout-aware, uses `HybridChunker` |
| Web crawling | [Crawl4AI](https://github.com/unclecode/crawl4ai) | Async crawler |
| Orchestration | [Prefect](https://prefect.io) | Local agent, flows in `graphrag/flows/` |
| API layer | [FastAPI](https://fastapi.tiangolo.com) | Query endpoint |

---

## Project Structure

```
graphrag/
├── models.py                    # Pydantic models: Document, Chunk, Entity, Relation
├── ingestion/
│   ├── pdf_loader.py            # Docling PDF → clean text + DoclingDocument cache
│   ├── web_crawler.py           # Crawl4AI async web crawler
│   └── chunker.py               # HybridChunker + paragraph-boundary fallback
├── extraction/
│   ├── entity_extractor.py      # Sarvam AI: text → entities + relations (harvest strategy)
│   └── entity_resolver.py       # Dedup entities via Qdrant cosine similarity
├── graph/
│   ├── schema.py                # Kuzu node/edge table definitions
│   └── store.py                 # Kuzu CRUD + Cypher query helpers
├── vector/
│   └── store.py                 # Qdrant collection management, upsert, search
├── query/
│   └── planner.py               # Fuses vector search + graph traversal → Sarvam answer
└── flows/
    └── ingest_flow.py           # Prefect flow: folder → graph + vectors

tests/
├── test_ingestion.py
├── test_extraction.py
└── test_query.py
```

---

## Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- A [Sarvam AI](https://sarvam.ai) API key

---

## Installation

**1. Clone the repo**
```bash
git clone https://github.com/SarangPratap/Graph_Rag.git
cd Graph_Rag
```

**2. Create and activate a virtual environment**
```bash
python -m venv graphrag-env

# Linux / macOS
source graphrag-env/bin/activate

# Windows
graphrag-env\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start Qdrant**
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

**5. Configure environment**
```bash
cp .env.example .env
# Edit .env and add your SARVAM_API_KEY
```

---

## Configuration

All configuration lives in `.env`:

```env
SARVAM_API_KEY=your_sarvam_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
KUZU_DB_PATH=./graphrag.db
```

The Kuzu database is embedded — no server setup required. It is created automatically on first run at the path specified by `KUZU_DB_PATH`.

---

## Usage

### Ingest a folder of PDFs

```python
from graphrag.flows.ingest_flow import ingest_flow

ingest_flow("./my_documents")
```

Or run the Prefect flow directly:
```bash
python -c "from graphrag.flows.ingest_flow import ingest_flow; ingest_flow('./my_documents')"
```

### Ask a question

```python
from graphrag.query.planner import answer_question
from graphrag.graph.store import get_connection
from graphrag.vector.store import get_client

answer = answer_question(
    question="How does GraphRAG improve multi-hop question answering?",
    qdrant_client=get_client(),
    kuzu_conn=get_connection(),
)
print(answer)
```

### Ingest a web page

```python
import asyncio
from graphrag.ingestion.web_crawler import crawl_url

doc = asyncio.run(crawl_url("https://example.com/article"))
```

---

## Knowledge Graph Schema

```
Nodes
─────
(:Document  {id, title, source, checksum, ingested_at})
(:Chunk     {id, text, source, doc_title})
(:Entity    {id, name, type, description})

Entity types: PERSON | ORG | CONCEPT | LOCATION | EVENT

Edges
─────
(:Document)-[:CONTAINS]->(:Chunk)
(:Chunk)-[:MENTIONS]->(:Entity)
(:Entity)-[:RELATES_TO {type, weight}]->(:Entity)
```

---

## Entity Extraction — How It Works

`sarvam-30b` is a reasoning model: its output lives in `reasoning_content`, not `content`. Asking it for a single top-level JSON block causes it to run out of tokens before finishing.

This project uses a **harvest strategy**:

1. The prompt instructs the model to emit one flat JSON object per entity or relation _as it thinks_.
2. The parser runs `re.finditer` over all `{...}` objects in `reasoning_content`.
3. Objects matching `{name, type, description}` become entities; objects matching `{source, target, relationship}` become relations.
4. Results are deduplicated by name (the model drafts its list multiple times during reasoning).

---

## Build Status

| Phase | Module | Status |
|---|---|---|
| 1 | `ingestion/pdf_loader.py` | ✅ Complete |
| 2 | `ingestion/chunker.py` | ✅ Complete |
| 3 | `extraction/entity_extractor.py` | ✅ Complete |
| 4 | `graph/schema.py` + `graph/store.py` | 🔲 In progress |
| 5 | `vector/store.py` | 🔲 Pending |
| 6 | `flows/ingest_flow.py` | 🔲 Pending |
| 7 | `ingestion/web_crawler.py` | 🔲 Pending |
| 8 | `query/planner.py` | 🔲 Pending |
| 9 | FastAPI endpoint | 🔲 Pending |

---

## Running Tests

```bash
pytest tests/
```

For a quick manual extraction test against the live API:
```bash
python tests/probe_sarvam.py
```

---

## Contributing

1. Fork the repo and create a feature branch: `git checkout -b feat/your-feature`
2. Follow the code standards in `CLAUDE.md` — type hints, module docstrings, `logging` over `print`
3. Run `pytest` before opening a PR
4. Open a pull request with a clear description of what changed and why

---

## License

MIT — see [LICENSE](LICENSE) for details.
