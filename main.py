"""FastAPI + Gradio application for the GraphRAG pipeline."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Generator

import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from graphrag.flows.ingest_flow import (
    chunk_documents,
    extract_entities,
    load_documents,
    store_in_graph,
    store_in_vector,
    ingest_flow,
)
from graphrag.graph.store import get_connection
from graphrag.query.planner import answer_question
from graphrag.vector.store import ensure_collections, get_client

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qdrant client is thread-safe and shared.
# Kuzu connections are NOT thread-safe — every callback calls get_connection() itself.
_qdrant = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources once on startup."""
    global _qdrant
    logger.info("Starting GraphRAG API...")
    _qdrant = get_client()
    get_connection()          # triggers schema creation on first open
    ensure_collections(_qdrant)
    logger.info("GraphRAG API ready.")
    yield
    logger.info("GraphRAG API shutting down.")


app = FastAPI(
    title="GraphRAG API",
    description="Graph-enhanced RAG: answer questions over your document collection.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Pydantic models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    """Return service liveness."""
    return {"status": "ok"}


@app.post("/ingest/upload")
async def ingest_upload(files: list[UploadFile] = File(...)):
    """Save uploaded PDFs to a temp folder and run the ingest pipeline."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            dest = os.path.join(tmpdir, f.filename or "upload.pdf")
            with open(dest, "wb") as out:
                out.write(await f.read())
        ingest_flow(tmpdir)
    return {"message": f"Ingested {len(files)} file(s) successfully."}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Answer a question using the GraphRAG pipeline."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    if _qdrant is None:
        raise HTTPException(status_code=503, detail="Server is still initialising.")
    try:
        ans = answer_question(question, _qdrant, get_connection())
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return QueryResponse(question=question, answer=ans)


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _get_stats() -> tuple[str, str, str]:
    """Query Kuzu for document, chunk, and entity counts."""
    try:
        conn = get_connection()
        docs     = conn.execute("MATCH (d:Document) RETURN count(d)").get_next()[0]
        chunks   = conn.execute("MATCH (c:Chunk)    RETURN count(c)").get_next()[0]
        entities = conn.execute("MATCH (e:Entity)   RETURN count(e)").get_next()[0]
        return str(int(docs)), str(int(chunks)), str(int(entities))
    except Exception as exc:
        logger.warning("Stats query failed: %s", exc)
        return "—", "—", "—"


def _ui_ingest(files) -> Generator[tuple, None, None]:
    """Ingest uploaded PDFs with granular progress updates.

    Yields (status_text, file_component_update) tuples so the file input
    can be cleared on success and left intact on failure for easy retry.
    """
    _keep = gr.update()   # sentinel: leave the file component unchanged

    if not files:
        yield "No files selected. Please upload at least one PDF.", _keep
        return

    n = len(files)

    # ── Step 1: save to temp dir ───────────────────────────────────────────────
    yield f"[1/5] Saving {n} file(s)...", _keep
    try:
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
        for f in files:
            src = f if isinstance(f, str) else f.name
            shutil.copy(src, tmpdir)
    except Exception as exc:
        logger.error("File save failed: %s", exc)
        yield f"Failed to save uploaded files: {exc}\n\nPlease try again.", _keep
        return

    try:
        # ── Step 2: load PDFs ──────────────────────────────────────────────────
        yield f"[2/5] Parsing {n} PDF(s)...", _keep
        docs = load_documents(tmpdir)

        if not docs:
            yield (
                "None of the uploaded files could be parsed as valid PDFs.\n"
                "Make sure the files are not password-protected or corrupted, then try again.",
                _keep,
            )
            return

        skipped = n - len(docs)
        skip_note = f"  ({skipped} file(s) could not be parsed and were skipped)" if skipped else ""
        yield f"[2/5] Parsed {len(docs)} document(s).{skip_note}\n[3/5] Chunking...", _keep

        # ── Step 3: chunk ──────────────────────────────────────────────────────
        chunks = chunk_documents(docs)
        if not chunks:
            yield (
                "Documents were loaded but produced no text chunks.\n"
                "The PDFs may be image-only scans. Try OCR-processed PDFs.",
                _keep,
            )
            return

        yield (
            f"[3/5] Created {len(chunks)} chunk(s).\n"
            f"[4/5] Extracting entities via Sarvam AI — {len(chunks)} chunk(s), "
            "5 in parallel. Check the terminal for per-chunk progress...",
            _keep,
        )

        # ── Step 4: extract entities ───────────────────────────────────────────
        results = extract_entities(chunks)
        total_entities  = sum(len(r.entities)  for r in results)
        total_relations = sum(len(r.relations) for r in results)

        if total_entities == 0:
            yield (
                f"[4/5] Extraction returned 0 entities across {len(chunks)} chunks.\n"
                "This may indicate a Sarvam API issue. The documents will still be stored "
                "for vector search, but graph traversal will not work.\n"
                "[5/5] Storing documents...",
                _keep,
            )
        else:
            yield (
                f"[4/5] Extracted {total_entities} entities and {total_relations} relations.\n"
                "[5/5] Writing to knowledge graph and vector store...",
                _keep,
            )

        # ── Step 5: store ──────────────────────────────────────────────────────
        store_in_graph(docs, chunks, results)
        store_in_vector(chunks, results)

        summary = (
            f"Ingestion complete!\n"
            f"  Documents : {len(docs)}\n"
            f"  Chunks    : {len(chunks)}\n"
            f"  Entities  : {total_entities}\n"
            f"  Relations : {total_relations}\n"
            f"{('  Warning: ' + str(skipped) + ' file(s) skipped — could not be parsed.') if skipped else ''}"
        ).strip()

        # Clear the file input on success so the user cannot accidentally re-ingest
        yield summary, None

    except Exception as exc:
        logger.error("Ingestion pipeline failed: %s", exc, exc_info=True)
        yield (
            f"Ingestion failed at an unexpected step:\n{exc}\n\n"
            "Your uploaded files are still selected — fix the issue and click Run Ingestion again.",
            _keep,
        )
    finally:
        tmpdir_obj.cleanup()


def _ui_chat(message: str, history: list) -> tuple[list, str]:
    """Answer a question and append to chat history."""
    message = message.strip()
    if not message:
        return history or [], ""

    if _qdrant is None:
        answer = "The server is still initialising — please wait a moment and try again."
    else:
        try:
            answer = answer_question(message, _qdrant, get_connection())
        except Exception as exc:
            logger.error("Query error: %s", exc)
            answer = (
                "Something went wrong while generating the answer. "
                "Check that documents have been ingested and try again."
            )

    updated = (history or []) + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": answer},
    ]
    return updated, ""


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="GraphRAG — Knowledge Graph Q&A") as _gradio_app:

    gr.Markdown("""
# GraphRAG — Knowledge Graph Q&A
Upload documents → build a knowledge graph → ask complex questions that require connecting information across multiple sources.
""")

    with gr.Tab("Knowledge Base"):

        gr.Markdown("### Graph Statistics")
        with gr.Row():
            stat_docs     = gr.Textbox(value="—", label="Documents",  interactive=False)
            stat_chunks   = gr.Textbox(value="—", label="Chunks",     interactive=False)
            stat_entities = gr.Textbox(value="—", label="Entities",   interactive=False)
        refresh_btn = gr.Button("Refresh Stats", size="sm")

        gr.Markdown("---")
        gr.Markdown("### Upload & Ingest")
        gr.Markdown(
            "Upload one or more PDFs. The pipeline will parse them, extract entities and "
            "relationships, build the knowledge graph, and index embeddings for vector search."
        )

        pdf_input = gr.File(
            file_count="multiple",
            file_types=[".pdf"],
            label="PDF files",
        )
        ingest_btn    = gr.Button("Run Ingestion", variant="primary")
        ingest_status = gr.Textbox(
            label="Ingestion Log",
            interactive=False,
            lines=6,
            placeholder="Status will appear here once you click Run Ingestion...",
        )

        _gradio_app.load(_get_stats, outputs=[stat_docs, stat_chunks, stat_entities])
        refresh_btn.click(_get_stats, outputs=[stat_docs, stat_chunks, stat_entities])
        ingest_btn.click(
            _ui_ingest,
            inputs=pdf_input,
            outputs=[ingest_status, pdf_input],
        ).then(
            _get_stats, outputs=[stat_docs, stat_chunks, stat_entities]
        )

    with gr.Tab("Ask Questions"):

        gr.Markdown(
            "Ask anything about the documents you have ingested. "
            "GraphRAG uses vector search + graph traversal to find the most accurate answer."
        )

        chatbot = gr.Chatbot(
            label="Conversation",
            height=500,
            placeholder="Answers will appear here after you ask a question.",
            buttons=["copy"],
            layout="bubble",
        )
        with gr.Row():
            question_input = gr.Textbox(
                placeholder="Ask a question about your documents...",
                show_label=False,
                scale=9,
                container=False,
            )
            ask_btn = gr.Button("Ask", variant="primary", scale=1)
        clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

        ask_btn.click(
            _ui_chat,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input],
        )
        question_input.submit(
            _ui_chat,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input],
        )
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, question_input])


app = gr.mount_gradio_app(app, _gradio_app, path="/")


if __name__ == "__main__":
    import uvicorn
    PORT = 7860
    print(f"""
╔══════════════════════════════════════════════╗
║          GraphRAG — Knowledge Graph Q&A      ║
╠══════════════════════════════════════════════╣
║  UI  →  http://localhost:{PORT}                ║
║  API →  http://localhost:{PORT}/docs           ║
╚══════════════════════════════════════════════╝
""")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
