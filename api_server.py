"""HTTP API for the ASTRION premium web UI. Never expose API keys."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    DOCUMENTS_DIR,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL,
    METADATA_PATH,
    PROJECT_ROOT,
)
from src.generate import EMPTY_GROUNDED_ANSWER, LOW_RELEVANCE_ANSWER, redact_secrets
from src.ingest import IngestionError, run_ingestion
from src.rag import RAGPipeline, configure_logging

load_dotenv(PROJECT_ROOT / ".env")
configure_logging()

app = FastAPI(title="ASTRION RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: RAGPipeline | None = None
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline.load()
    return _pipeline


def reset_pipeline() -> None:
    """Drop cached retriever so Ask uses the latest FAISS index."""
    global _pipeline
    _pipeline = None


def _safe_pdf_filename(filename: str | None) -> str:
    raw = Path(filename or "document.pdf").name.strip()
    stem = Path(raw).stem
    cleaned = _SAFE_NAME.sub("_", stem).strip("._") or "document"
    return f"{cleaned[:120]}.pdf"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


def _index_stats() -> dict[str, Any]:
    if not METADATA_PATH.is_file():
        return {
            "ready": False,
            "documents": 0,
            "chunks": 0,
            "pages": 0,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "llm_model": GROQ_MODEL,
            "vector_store": "FAISS",
            "documents_list": [],
        }

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    vectors = metadata.get("vectors") or []
    by_source: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"pages": set(), "chunks": 0}
    )
    for item in vectors:
        source = str(item.get("source") or "unknown")
        by_source[source]["chunks"] += 1
        page = item.get("page")
        if page is not None:
            by_source[source]["pages"].add(int(page))

    documents_list = []
    for source, stats in sorted(by_source.items(), key=lambda kv: kv[0].lower()):
        documents_list.append(
            {
                "id": source,
                "filename": Path(source).name,
                "pages": len(stats["pages"]),
                "chunks": stats["chunks"],
                "status": "Indexed",
            }
        )

    return {
        "ready": True,
        "documents": len(documents_list),
        "chunks": int(metadata.get("num_vectors") or len(vectors)),
        "pages": sum(doc["pages"] for doc in documents_list),
        "embedding_model": str(metadata.get("embedding_model") or EMBEDDING_MODEL_NAME),
        "llm_model": GROQ_MODEL,
        "vector_store": "FAISS",
        "documents_list": documents_list,
        "documents_dir": str(DOCUMENTS_DIR),
    }


def _classify_result(result: dict[str, Any]) -> str:
    answer = str(result.get("answer") or "")
    if result.get("retrieval_count", 0) == 0 and answer in {
        EMPTY_GROUNDED_ANSWER,
        LOW_RELEVANCE_ANSWER,
    }:
        return "no-results"
    lowered = answer.lower()
    failure_tokens = (
        "timed out",
        "language model request failed",
        "language model returned",
        "groq_api_key is missing",
        "rejected the api key",
        "rate-limited",
        "invalid environment configuration",
        "faiss index is missing",
        "metadata is missing",
        "could not embed",
        "unexpected error",
    )
    if any(token in lowered for token in failure_tokens):
        return "error"
    return "answer"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "astrion-rag"}


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    from src.generate import groq_is_configured

    payload = _index_stats()
    payload["groq_ready"] = groq_is_configured()
    return payload


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    payload = _index_stats()
    return {
        "documents": payload["documents_list"],
        "total_documents": payload["documents"],
        "total_pages": payload["pages"],
        "total_chunks": payload["chunks"],
        "ready": payload["ready"],
        "documents_dir": payload.get("documents_dir"),
    }


@app.post("/api/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one PDF.")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for upload in files:
        original = upload.filename or "document.pdf"
        if not original.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are supported ({original}).",
            )
        name = _safe_pdf_filename(original)
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"{name} is empty.")
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"{name} exceeds the 25 MB upload limit.",
            )
        if not data.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail=f"{name} is not a valid PDF.")

        (DOCUMENTS_DIR / name).write_bytes(data)
        saved.append(name)

    try:
        result = run_ingestion()
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail="Indexing failed. Please try again.",
        ) from exc

    reset_pipeline()
    payload = _index_stats()
    return {
        "saved": saved,
        "message": f"Indexed {len(saved)} PDF{'s' if len(saved) != 1 else ''}.",
        "ingestion": {
            "documents_processed": result.documents_processed,
            "chunks": result.chunks_created,
            "pages": result.pages_processed,
        },
        "documents": payload["documents_list"],
        "total_documents": payload["documents"],
        "total_pages": payload["pages"],
        "total_chunks": payload["chunks"],
        "ready": payload["ready"],
        "documents_dir": payload.get("documents_dir"),
    }


@app.post("/api/ask")
def ask(body: AskRequest) -> dict[str, Any]:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    try:
        result = get_pipeline().answer(question)
    except Exception as exc:  # noqa: BLE001
        return {
            "state": "error",
            "question": question,
            "answer": "Something went wrong while answering. Please try again.",
            "sources": [],
            "retrieved_chunks": [],
            "retrieval_count": 0,
            "latency": {
                "retrieval_seconds": 0.0,
                "generation_seconds": 0.0,
                "total_seconds": 0.0,
            },
            "error": redact_secrets(f"{type(exc).__name__}"),
        }

    answer = redact_secrets(str(result.get("answer") or ""))
    state = _classify_result({**result, "answer": answer})
    return {
        "state": state,
        "question": question,
        "answer": answer,
        "sources": result.get("sources") or [],
        "retrieved_chunks": result.get("retrieved_chunks") or [],
        "retrieval_count": result.get("retrieval_count") or 0,
        "latency": result.get("latency") or {},
    }


@app.get("/api/evaluation")
def evaluation() -> dict[str, Any]:
    report_path = PROJECT_ROOT / "evaluation" / "report.json"
    results_path = PROJECT_ROOT / "evaluation" / "results.json"
    if not report_path.is_file():
        return {
            "available": False,
            "message": "No evaluation report yet. Run: python -m evaluation.evaluate",
            "report": None,
            "rows": [],
            "failures": [],
        }

    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if results_path.is_file():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        for item in payload.get("results") or []:
            retrieved = item.get("retrieved_sources") or []
            expected = item.get("expected_sources") or []
            retrieved_name = (
                retrieved[0].get("source")
                if retrieved and isinstance(retrieved[0], dict)
                else "—"
            )
            expected_name = expected[0] if expected else "—"
            hit = item.get("retrieval_hit")
            if hit is True:
                outcome = "Pass"
            elif hit is False:
                outcome = "Fail"
            else:
                outcome = "N/A"
            latency = item.get("total_latency")
            rows.append(
                {
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "expected_source": expected_name,
                    "retrieved_source": retrieved_name,
                    "result": outcome,
                    "latency": None if latency is None else f"{float(latency):.2f}s",
                    "status": item.get("status"),
                }
            )
        for item in payload.get("failures") or []:
            failures.append(
                {
                    "id": item.get("id"),
                    "title": item.get("question") or item.get("id"),
                    "description": redact_secrets(str(item.get("error") or "Failed")),
                    "cause": "Pipeline or API failure during evaluation.",
                    "mitigation": "Fix environment/API configuration and re-run evaluation.",
                }
            )

    hit_rate = report.get("retrieval_hit_rate")
    return {
        "available": True,
        "report": report,
        "metrics": [
            {
                "label": "Questions Evaluated",
                "value": str(report.get("total_questions") or 0),
                "unit": "",
            },
            {
                "label": "Completed",
                "value": str(report.get("completed") or 0),
                "unit": "",
            },
            {
                "label": "Failed",
                "value": str(report.get("failed") or 0),
                "unit": "",
            },
            {
                "label": "Retrieval Hit Rate",
                "value": "—" if hit_rate is None else f"{float(hit_rate) * 100:.1f}",
                "unit": "" if hit_rate is None else "%",
            },
            {
                "label": "Average Latency",
                "value": (
                    "—"
                    if report.get("average_latency") is None
                    else f"{float(report['average_latency']):.2f}"
                ),
                "unit": "" if report.get("average_latency") is None else "s",
            },
        ],
        "rows": rows,
        "failures": failures,
        "groundedness": report.get("groundedness"),
        "citation_score": report.get("citation_score"),
        "notes": report.get("notes"),
    }


def main() -> None:
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
