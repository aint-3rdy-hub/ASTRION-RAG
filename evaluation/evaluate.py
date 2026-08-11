"""Run the RAG pipeline over evaluation/eval_dataset.json.

Retrieval hit/miss is automatic (expected source overlap).
Groundedness and citation quality are left null for manual scoring.
API failures are recorded and skipped; they do not stop the run.
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.config import PROJECT_ROOT
from src.generate import redact_secrets
from src.rag import RAGPipeline, configure_logging

DATASET_PATH = PROJECT_ROOT / "evaluation" / "eval_dataset.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results.json"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "report.json"

_API_FAILURE_MARKERS = (
    "timed out",
    "language model request failed",
    "language model returned",
    "groq_api_key is missing",
    "rejected the api key",
    "rate-limited",
    "invalid environment configuration",
    "could not embed the question",
    "unexpected error occurred",
)


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError(f"Evaluation dataset has no items: {path}")
    return items


def _normalize_source(value: str) -> str:
    return Path(str(value).replace("\\", "/")).name.strip().lower()


def expected_source_names(expected_sources: list[Any]) -> list[str]:
    names: list[str] = []
    for item in expected_sources:
        if isinstance(item, str):
            names.append(_normalize_source(item))
        elif isinstance(item, dict) and item.get("source"):
            names.append(_normalize_source(str(item["source"])))
    return names


def retrieval_hit(retrieved_sources: list[Any], expected_sources: list[Any]) -> bool | None:
    """True if at least one expected source was retrieved.

    Items with no expected sources are not scored (manual / no-evidence cases).
    """
    expected = expected_source_names(expected_sources)
    if not expected:
        return None
    retrieved = {
        _normalize_source(str(item.get("source", "")))
        for item in retrieved_sources
        if isinstance(item, dict)
    }
    return any(name in retrieved for name in expected)


def _is_api_failure(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _API_FAILURE_MARKERS)


def evaluate_item(pipeline: RAGPipeline, item: dict[str, Any]) -> dict[str, Any]:
    question_id = str(item.get("id") or "")
    question = str(item.get("question") or "").strip()
    expected_answer = item.get("expected_answer")
    expected_sources = list(item.get("expected_sources") or [])

    try:
        result = pipeline.answer(question)
    except Exception as exc:  # noqa: BLE001
        return {
            "id": question_id,
            "question": question,
            "status": "failed",
            "error": redact_secrets(f"{type(exc).__name__}: {exc}"),
            "answer": None,
            "expected_answer": expected_answer,
            "retrieved_sources": [],
            "expected_sources": expected_sources,
            "retrieval_hit": None,
            "total_latency": None,
            "groundedness": None,
            "citation_score": None,
        }

    answer = redact_secrets(str(result.get("answer") or ""))
    retrieved_sources = list(result.get("sources") or [])
    latency = (result.get("latency") or {}).get("total_seconds")
    if _is_api_failure(answer):
        return {
            "id": question_id,
            "question": question,
            "status": "failed",
            "error": answer,
            "answer": answer,
            "expected_answer": expected_answer,
            "retrieved_sources": retrieved_sources,
            "expected_sources": expected_sources,
            "retrieval_hit": retrieval_hit(retrieved_sources, expected_sources),
            "total_latency": latency,
            "latency": result.get("latency"),
            "retrieval_count": result.get("retrieval_count"),
            "groundedness": None,
            "citation_score": None,
        }

    return {
        "id": question_id,
        "question": question,
        "status": "ok",
        "answer": answer,
        "expected_answer": expected_answer,
        "retrieved_sources": retrieved_sources,
        "expected_sources": expected_sources,
        "retrieval_hit": retrieval_hit(retrieved_sources, expected_sources),
        "total_latency": latency,
        "latency": result.get("latency"),
        "retrieval_count": result.get("retrieval_count"),
        "retrieved_chunks": result.get("retrieved_chunks") or [],
        "groundedness": None,
        "citation_score": None,
    }


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in records if row.get("status") == "ok"]
    failed = [row for row in records if row.get("status") == "failed"]
    scored = [row for row in completed if row.get("retrieval_hit") is not None]
    hits = [row for row in scored if row.get("retrieval_hit") is True]
    latencies = [
        float(row["total_latency"])
        for row in completed
        if isinstance(row.get("total_latency"), (int, float))
    ]
    hit_rate = (len(hits) / len(scored)) if scored else None
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_questions": len(records),
        "completed": len(completed),
        "failed": len(failed),
        "failed_question_ids": [row.get("id") for row in failed],
        "retrieval_questions_scored": len(scored),
        "retrieval_hits": len(hits),
        "retrieval_hit_rate": None if hit_rate is None else round(hit_rate, 4),
        "average_latency": None if not latencies else round(statistics.mean(latencies), 4),
        "minimum_latency": None if not latencies else round(min(latencies), 4),
        "maximum_latency": None if not latencies else round(max(latencies), 4),
        "groundedness": None,
        "citation_score": None,
        "notes": (
            "groundedness and citation_score are reserved for manual evaluation. "
            "retrieval_hit_rate counts items with non-empty expected_sources where "
            "at least one expected filename appears in retrieved sources."
        ),
    }


def format_summary(report: dict[str, Any], results_path: Path, report_path: Path) -> str:
    hit_rate = report.get("retrieval_hit_rate")
    hit_display = "n/a" if hit_rate is None else f"{hit_rate * 100:.1f}%"
    avg = report.get("average_latency")
    mn = report.get("minimum_latency")
    mx = report.get("maximum_latency")
    failed_ids = report.get("failed_question_ids") or []
    lines = [
        "============================================================",
        "ASTRION RAG - EVALUATION",
        "============================================================",
        "",
        f"Total questions:           {report.get('total_questions')}",
        f"Completed:                 {report.get('completed')}",
        f"Failed:                    {report.get('failed')}",
        f"Retrieval hit rate:        {hit_display} ({report.get('retrieval_hits')}/{report.get('retrieval_questions_scored')})",
        f"Average latency:           {avg if avg is not None else 'n/a'} seconds",
        f"Minimum latency:           {mn if mn is not None else 'n/a'} seconds",
        f"Maximum latency:           {mx if mx is not None else 'n/a'} seconds",
        "",
        "Manual fields (not auto-scored):",
        "  groundedness:             null",
        "  citation_score:           null",
        "",
        "Results:",
        str(results_path),
        "",
        "Report:",
        str(report_path),
    ]
    if failed_ids:
        lines.extend(["", "Failed question IDs:", "  " + ", ".join(str(i) for i in failed_ids)])
    lines.extend(
        [
            "",
            "============================================================",
            "EVALUATION COMPLETE",
            "============================================================",
        ]
    )
    return "\n".join(lines)


def run_evaluation(
    pipeline: RAGPipeline | None = None,
    dataset_path: Path = DATASET_PATH,
    results_path: Path = RESULTS_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    items = load_dataset(dataset_path)
    if pipeline is None:
        pipeline = RAGPipeline.load()

    records = [evaluate_item(pipeline, item) for item in items]
    report = build_report(records)
    results_payload = {
        "dataset": str(dataset_path),
        "report": report,
        "results": [row for row in records if row.get("status") == "ok"],
        "failures": [row for row in records if row.get("status") == "failed"],
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_summary(report, results_path, report_path))
    return results_payload


def main() -> int:
    configure_logging()
    try:
        run_evaluation()
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
