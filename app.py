"""NEXUS-RAG Streamlit demo. Never display API keys or stack traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

import streamlit as st

from src.config import EMBEDDING_MODEL_NAME, GROQ_MODEL, METADATA_PATH
from src.generate import EMPTY_GROUNDED_ANSWER, LOW_RELEVANCE_ANSWER, redact_secrets
from src.rag import RAGPipeline, configure_logging

configure_logging()

st.set_page_config(page_title="NEXUS-RAG", layout="centered")


def _index_stats() -> dict[str, Any]:
    if not METADATA_PATH.is_file():
        return {
            "documents": 0,
            "chunks": 0,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "ready": False,
        }
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    vectors = metadata.get("vectors") or []
    sources = {str(item.get("source", "")) for item in vectors if item.get("source")}
    return {
        "documents": len(sources),
        "chunks": int(metadata.get("num_vectors") or len(vectors)),
        "embedding_model": str(metadata.get("embedding_model") or EMBEDDING_MODEL_NAME),
        "ready": True,
    }


@st.cache_resource(show_spinner="Loading the document index…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.load()


def _is_empty_retrieval(result: dict[str, Any]) -> bool:
    answer = result.get("answer")
    return result.get("retrieval_count", 0) == 0 and answer in {
        EMPTY_GROUNDED_ANSWER,
        LOW_RELEVANCE_ANSWER,
    }


def _is_groq_failure(answer: str) -> bool:
    lowered = answer.lower()
    return any(
        token in lowered
        for token in (
            "timed out",
            "language model request failed",
            "language model returned",
            "groq_api_key is missing",
            "rejected the api key",
            "rate-limited",
            "invalid environment configuration",
        )
    )


def _is_system_failure(answer: str) -> bool:
    lowered = answer.lower()
    return any(
        token in lowered
        for token in (
            "faiss index is missing",
            "metadata is missing",
            "could not embed",
            "unexpected error",
        )
    )


def render_result(result: dict[str, Any]) -> None:
    answer = redact_secrets(str(result.get("answer") or ""))
    latency = result.get("latency") or {}
    sources = result.get("sources") or []
    chunks = result.get("retrieved_chunks") or []

    if _is_empty_retrieval(result):
        if answer == LOW_RELEVANCE_ANSWER:
            st.warning(
                "Retrieved passages were below the relevance threshold, so no answer "
                "was generated from them."
            )
        else:
            st.warning(
                "The system could not find sufficient evidence in the indexed documents "
                "to answer this question."
            )
        st.write(answer)
    elif _is_groq_failure(answer):
        st.error(answer)
    elif _is_system_failure(answer):
        st.error(
            "The document index is not ready. Add PDFs to data/documents/ and run "
            "ingestion before asking questions."
        )
    else:
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        if sources:
            for item in sources:
                st.markdown(
                    f"- **{item.get('source', 'unknown')}**, "
                    f"page {item.get('page', '?')} "
                    f"(`{item.get('chunk_id', '')}`)"
                )
        else:
            st.caption("No source citations were returned.")

    cols = st.columns(3)
    cols[0].metric("Retrieval", f"{latency.get('retrieval_seconds', 0):.2f}s")
    cols[1].metric("Generation", f"{latency.get('generation_seconds', 0):.2f}s")
    cols[2].metric("Total", f"{latency.get('total_seconds', 0):.2f}s")

    with st.expander("Retrieved chunks"):
        if not chunks:
            st.caption("No chunks were retrieved.")
        for chunk in chunks:
            st.markdown(
                f"**{chunk.get('source', '')}** · page {chunk.get('page', '?')} · "
                f"`{chunk.get('chunk_id', '')}`"
            )
            st.write(chunk.get("text", ""))
            st.divider()

    with st.expander("Retrieval Trace"):
        if not chunks:
            st.caption("No retrieval trace (empty or failed retrieval).")
        for chunk in chunks:
            st.markdown(
                f"- **source:** {chunk.get('source', '')}  \n"
                f"  **page:** {chunk.get('page', '?')}  \n"
                f"  **chunk ID:** `{chunk.get('chunk_id', '')}`  \n"
                f"  **relevance score:** {chunk.get('score', 0):.4f}"
            )


def main() -> None:
    st.title("NEXUS-RAG")
    st.caption("Citation-Grounded Document Intelligence")

    stats = _index_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indexed documents", stats["documents"])
    c2.metric("Indexed chunks", stats["chunks"])
    c3.metric("Embedding model", stats["embedding_model"])
    c4.metric("LLM model", GROQ_MODEL)

    if not stats["ready"]:
        st.info(
            "No index found. Place PDFs in `data/documents/` and run "
            "`python -m src.ingest` from the project virtual environment."
        )

    with st.form("ask"):
        question = st.text_input("Question", placeholder="Ask a question about your documents")
        submitted = st.form_submit_button("Ask", type="primary")

    if not submitted:
        return
    if not question.strip():
        st.warning("Enter a question to continue.")
        return

    try:
        pipeline = get_pipeline()
        with st.spinner("Retrieving evidence and generating an answer…"):
            result = pipeline.answer(question.strip())
    except Exception:  # noqa: BLE001
        st.error(
            "Something went wrong while answering. "
            "Please try again. No internal details are shown here."
        )
        return

    render_result(result)


main()
