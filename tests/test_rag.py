"""Focused tests for ASTRION RAG document ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.chunking import chunk_pages, make_chunk_id, split_into_word_chunks
from src.cleaning import clean_text
from src.documents import discover_pdfs, extract_pdf, list_unsupported_files, load_documents
from src.indexer import build_metadata, save_index
from src.ingest import IngestionError, run_ingestion
from src.models import Chunk, ExtractedPage
from src.generate import (
    EMPTY_GROUNDED_ANSWER,
    LOW_RELEVANCE_ANSWER,
    GenerationError,
    Generator,
    redact_secrets,
)
from src.rag import RAGPipeline
from src.retrieve import RetrievalError, Retriever, retrieve_sample
from tests.pdf_utils import write_simple_pdf


def test_clean_text_collapses_whitespace_and_blank_lines() -> None:
    raw = "Hello   world\tfrom\nPDF.\n\n\n\nNext   paragraph."
    cleaned = clean_text(raw)
    assert cleaned == "Hello world from PDF.\n\nNext paragraph."


def test_clean_text_joins_unnecessary_line_breaks() -> None:
    raw = "This sentence is split\nacross wrapped PDF lines.\n\nA new paragraph."
    cleaned = clean_text(raw)
    assert cleaned == (
        "This sentence is split across wrapped PDF lines.\n\nA new paragraph."
    )


def test_clean_text_empty_input() -> None:
    assert clean_text("") == ""
    assert clean_text("   \n\n  ") == ""


def test_chunk_generation_word_windows() -> None:
    words = [f"word{i:04d}" for i in range(1000)]
    text = " ".join(words)
    chunks = split_into_word_chunks(text, chunk_size=800, chunk_overlap=120)

    assert len(chunks) == 2
    assert chunks[0].split() == words[:800]
    assert chunks[1].split() == words[680:]


def test_empty_page_still_produces_a_chunk() -> None:
    pages = [ExtractedPage(source="blank.pdf", page=1, text="")]
    chunks = chunk_pages(pages, chunk_size=800, chunk_overlap=120)
    assert len(chunks) == 1
    assert chunks[0].text == ""
    assert chunks[0].chunk_id == "blank_p1_c01"


def test_chunk_ids_are_deterministic() -> None:
    pages = [
        ExtractedPage(
            source="network_security.pdf",
            page=12,
            text="alpha " * 50 + "beta " * 50,
        )
    ]
    first = chunk_pages(pages)
    second = chunk_pages(pages)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id == "network_security_p12_c01"
    assert make_chunk_id("network_security.pdf", 12, 3) == "network_security_p12_c03"


def test_discover_pdfs_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "documents"
    empty.mkdir()
    (empty / "notes.txt").write_text("not a pdf", encoding="utf-8")
    assert discover_pdfs(empty) == []


def test_discover_pdfs_missing_directory(tmp_path: Path) -> None:
    assert discover_pdfs(tmp_path / "missing") == []


def test_discover_pdfs_sorted_and_recursive(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    nested = documents / "nested"
    nested.mkdir(parents=True)
    write_simple_pdf(documents / "zeta.pdf", ["zeta"])
    write_simple_pdf(nested / "alpha.pdf", ["alpha"])
    write_simple_pdf(documents / "beta.pdf", ["beta"])
    found = discover_pdfs(documents)
    names = [path.relative_to(documents).as_posix() for path in found]
    assert names == ["beta.pdf", "nested/alpha.pdf", "zeta.pdf"]


def test_load_documents_records_invalid_pdf(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "broken.pdf").write_bytes(b"this is not a pdf")
    pages, failures, discovered = load_documents(documents)
    assert discovered == 1
    assert pages == []
    assert len(failures) == 1
    assert failures[0].path == "broken.pdf"
    assert "invalid" in failures[0].reason.lower() or "unreadable" in failures[0].reason.lower()


def test_extract_pdf_keeps_empty_pages(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    pdf_path = documents / "mixed.pdf"
    write_simple_pdf(pdf_path, ["Visible sentence on page one.", ""])
    pages = extract_pdf(pdf_path, documents)
    assert [page.page for page in pages] == [1, 2]
    assert "Visible sentence" in pages[0].text
    assert pages[1].text == ""


def test_metadata_structure_maps_vector_positions() -> None:
    chunks = [
        Chunk(chunk_id="doc_p1_c01", source="doc.pdf", page=1, text="first"),
        Chunk(chunk_id="doc_p1_c02", source="doc.pdf", page=1, text="second"),
    ]
    metadata = build_metadata(
        chunks,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
    )
    assert metadata["embedding_model"] == "all-MiniLM-L6-v2"
    assert metadata["embedding_dimension"] == 384
    assert metadata["num_vectors"] == 2
    assert metadata["chunk_unit"] == "words"
    assert [item["vector_id"] for item in metadata["vectors"]] == [0, 1]
    assert metadata["vectors"][0]["chunk_id"] == "doc_p1_c01"
    assert metadata["vectors"][1]["text"] == "second"
    for item in metadata["vectors"]:
        assert set(item) >= {"vector_id", "chunk_id", "source", "page", "text"}


def test_save_index_writes_matching_metadata(tmp_path: Path) -> None:
    chunks = [
        Chunk(chunk_id="doc_p1_c01", source="doc.pdf", page=1, text="alpha"),
        Chunk(chunk_id="doc_p2_c01", source="doc.pdf", page=2, text="beta"),
    ]
    embeddings = np.zeros((2, 8), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    index_dir = tmp_path / "index"
    metadata = save_index(
        embeddings,
        chunks,
        index_path=index_dir / "faiss.index",
        metadata_path=index_dir / "metadata.json",
        embedding_model="stub-model",
    )
    saved = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    assert saved["num_vectors"] == 2
    assert saved["vectors"][0]["chunk_id"] == chunks[0].chunk_id
    assert (index_dir / "faiss.index").is_file()
    assert metadata["num_vectors"] == len(saved["vectors"])


def test_ingestion_missing_documents_directory(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="Documents directory is missing"):
        run_ingestion(
            documents_dir=tmp_path / "missing-docs",
            index_dir=tmp_path / "index",
        )


def test_list_unsupported_files_ignores_non_pdfs(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "notes.txt").write_text("not a pdf", encoding="utf-8")
    (documents / ".gitkeep").write_text("", encoding="utf-8")
    write_simple_pdf(documents / "keep.pdf", ["keep me"])
    assert list_unsupported_files(documents) == ["notes.txt"]


def test_ingestion_reports_unsupported_and_empty_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "readme.txt").write_text("ignored", encoding="utf-8")
    write_simple_pdf(documents / "empty.pdf", [""])
    write_simple_pdf(documents / "usable.pdf", ["Searchable policy text."])

    class StubModel:
        def encode(self, texts, **kwargs):  # noqa: ANN001
            return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr("src.ingest.get_embedding_model", lambda name: StubModel())
    result = run_ingestion(
        documents_dir=documents,
        index_dir=tmp_path / "index",
        embedding_model_name="stub-model",
    )
    assert "readme.txt" in result.unsupported_files
    assert result.documents_processed == 1
    assert result.chunks_created == 1
    assert any("Empty extracted text" in failure.reason for failure in result.failures)
    metadata = json.loads((tmp_path / "index" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["vectors"][0]["source"] == "usable.pdf"
    assert all(item["text"].strip() for item in metadata["vectors"])


def test_empty_document_directory_ingestion_skips_index(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    index_dir = tmp_path / "index"
    result = run_ingestion(documents_dir=documents, index_dir=index_dir)
    assert result.documents_discovered == 0
    assert result.pages_processed == 0
    assert result.chunks_created == 0
    assert result.skipped_index is True
    assert not (index_dir / "faiss.index").exists()


def test_ingestion_with_stub_embeddings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    documents = tmp_path / "documents"
    write_simple_pdf(
        documents / "network_security.pdf",
        ["Firewall rules belong in a documented security policy."],
    )

    class StubModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 4

        def encode(self, texts, **kwargs):  # noqa: ANN001
            matrix = np.zeros((len(texts), 4), dtype=np.float32)
            for i, _ in enumerate(texts):
                matrix[i, i % 4] = 1.0
            return matrix

    monkeypatch.setattr("src.ingest.get_embedding_model", lambda name: StubModel())
    index_dir = tmp_path / "index"
    result = run_ingestion(
        documents_dir=documents,
        index_dir=index_dir,
        embedding_model_name="stub-model",
    )
    assert result.documents_processed == 1
    assert result.pages_processed == 1
    assert result.chunks_created == 1
    assert result.embedding_dimension == 4
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["num_vectors"] == 1
    assert metadata["vectors"][0]["source"] == "network_security.pdf"
    assert metadata["vectors"][0]["chunk_id"] == "network_security_p1_c01"


def _write_stub_index(index_dir: Path) -> None:
    """3 orthonormal vectors so cosine scores are 1, 0, or -1."""
    import faiss

    vectors = np.eye(3, dtype=np.float32)
    index = faiss.IndexFlatIP(3)
    index.add(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    metadata = {
        "embedding_model": "stub-model",
        "embedding_dimension": 3,
        "num_vectors": 3,
        "vectors": [
            {
                "vector_id": 0,
                "chunk_id": "network_security_p1_c01",
                "source": "network_security.pdf",
                "page": 1,
                "text": "Firewall rules must match a written security policy.",
            },
            {
                "vector_id": 1,
                "chunk_id": "network_security_p2_c01",
                "source": "network_security.pdf",
                "page": 2,
                "text": "Prefer phishing-resistant authentication.",
            },
            {
                "vector_id": 2,
                "chunk_id": "processing_p1_c01",
                "source": "processing.pdf",
                "page": 1,
                "text": "Chunking uses overlapping windows.",
            },
        ],
    }
    (index_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


class _QueryStubModel:
    """Maps a query string to a 3-D unit vector used by the stub index."""

    mapping = {
        "firewall policy": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        "authentication": np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        "unrelated astronomy": np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
    }

    def encode(self, texts, **kwargs):  # noqa: ANN001
        text = texts[0]
        for key, vector in self.mapping.items():
            if key in text:
                return vector
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float32)


def test_retriever_missing_index_files(tmp_path: Path) -> None:
    with pytest.raises(RetrievalError, match="Index files are missing"):
        Retriever.load(
            index_path=tmp_path / "faiss.index",
            metadata_path=tmp_path / "metadata.json",
            model=_QueryStubModel(),
        )


def test_retriever_returns_ranked_hits(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    _write_stub_index(index_dir)
    retriever = Retriever.load(
        index_path=index_dir / "faiss.index",
        metadata_path=index_dir / "metadata.json",
        model=_QueryStubModel(),
        top_k=2,
        min_score=0.5,
    )
    result = retriever.search("firewall policy")
    assert result.empty is False
    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.rank == 1
    assert hit.score == pytest.approx(1.0)
    assert hit.source == "network_security.pdf"
    assert hit.page == 1
    assert hit.chunk_id == "network_security_p1_c01"
    assert "Firewall rules" in hit.text
    payload = result.to_dict()
    assert payload["hit_count"] == 1
    assert payload["hits"][0]["chunk_id"] == hit.chunk_id


def test_retriever_empty_when_below_threshold(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    _write_stub_index(index_dir)
    retriever = Retriever.load(
        index_path=index_dir / "faiss.index",
        metadata_path=index_dir / "metadata.json",
        model=_QueryStubModel(),
        top_k=5,
        min_score=0.5,
    )
    result = retriever.search("unrelated astronomy")
    assert result.empty is True
    assert result.hits == []
    assert result.empty_reason == "no_chunks_above_threshold"


def test_retriever_empty_query(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    _write_stub_index(index_dir)
    retriever = Retriever.load(
        index_path=index_dir / "faiss.index",
        metadata_path=index_dir / "metadata.json",
        model=_QueryStubModel(),
    )
    result = retriever.search("   ")
    assert result.empty is True
    assert result.empty_reason == "empty_query"


def test_retriever_top_k_and_deterministic_order(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    _write_stub_index(index_dir)
    retriever = Retriever.load(
        index_path=index_dir / "faiss.index",
        metadata_path=index_dir / "metadata.json",
        model=_QueryStubModel(),
        top_k=1,
        min_score=0.0,
    )
    first = retriever.search("authentication", top_k=1)
    second = retriever.search("authentication", top_k=1)
    assert [hit.chunk_id for hit in first.hits] == [hit.chunk_id for hit in second.hits]
    assert len(first.hits) == 1
    assert first.hits[0].chunk_id == "network_security_p2_c01"


def test_retrieve_sample_function(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_dir = tmp_path / "index"
    _write_stub_index(index_dir)
    monkeypatch.setattr("src.retrieve.FAISS_INDEX_PATH", index_dir / "faiss.index")
    monkeypatch.setattr("src.retrieve.METADATA_PATH", index_dir / "metadata.json")
    monkeypatch.setattr(
        "src.retrieve.get_embedding_model",
        lambda name: _QueryStubModel(),
    )
    result = retrieve_sample("firewall policy")
    assert result.empty is False
    assert result.hits[0].source == "network_security.pdf"


class _RecordingGenerator:
    def __init__(self, text: str = "Cited answer [network_security_p1_c01].") -> None:
        self.calls = 0
        self.text = text

    def generate(self, question, hits):  # noqa: ANN001
        self.calls += 1
        return self.text


class _FailingGenerator:
    def __init__(self, error: GenerationError) -> None:
        self.error = error

    def generate(self, question, hits):  # noqa: ANN001
        raise self.error


def _pipeline(tmp_path: Path, generator) -> RAGPipeline:  # noqa: ANN001
    index_dir = tmp_path / "index"
    _write_stub_index(index_dir)
    retriever = Retriever.load(
        index_path=index_dir / "faiss.index",
        metadata_path=index_dir / "metadata.json",
        model=_QueryStubModel(),
        top_k=5,
        min_score=0.5,
    )
    return RAGPipeline(retriever=retriever, generator=generator)


def test_rag_answer_success_structure(tmp_path: Path) -> None:
    generator = _RecordingGenerator()
    pipeline = _pipeline(tmp_path, generator)
    result = pipeline.answer("firewall policy")
    assert generator.calls == 1
    assert result["question"] == "firewall policy"
    assert result["answer"].startswith("Cited answer")
    assert result["retrieval_count"] == 1
    assert result["sources"][0]["source"] == "network_security.pdf"
    assert result["sources"][0]["page"] == 1
    assert result["retrieved_chunks"][0]["chunk_id"] == "network_security_p1_c01"
    assert set(result["latency"]) == {
        "retrieval_seconds",
        "generation_seconds",
        "total_seconds",
    }
    assert result["latency"]["generation_seconds"] >= 0


def test_rag_empty_retrieval_skips_llm(tmp_path: Path) -> None:
    generator = _RecordingGenerator()
    pipeline = _pipeline(tmp_path, generator)
    result = pipeline.answer("unrelated astronomy")
    assert generator.calls == 0
    assert result["answer"] == LOW_RELEVANCE_ANSWER
    assert result["retrieval_count"] == 0
    assert result["sources"] == []
    assert result["latency"]["generation_seconds"] == 0


def test_rag_missing_index(tmp_path: Path) -> None:
    pipeline = RAGPipeline(
        generator=_RecordingGenerator(),
        index_path=tmp_path / "missing" / "faiss.index",
        metadata_path=tmp_path / "missing" / "metadata.json",
        model=_QueryStubModel(),
    )
    result = pipeline.answer("firewall policy")
    assert "FAISS index is missing" in result["answer"]
    assert result["retrieval_count"] == 0


def test_rag_missing_metadata(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    _write_stub_index(index_dir)
    (index_dir / "metadata.json").unlink()
    pipeline = RAGPipeline(
        generator=_RecordingGenerator(),
        index_path=index_dir / "faiss.index",
        metadata_path=index_dir / "metadata.json",
        model=_QueryStubModel(),
    )
    result = pipeline.answer("firewall policy")
    assert "metadata" in result["answer"].lower()


def test_rag_groq_timeout(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        _FailingGenerator(GenerationError("timeout", timeout=True)),
    )
    result = pipeline.answer("firewall policy")
    assert "timed out" in result["answer"].lower()


def test_rag_groq_api_failure(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        _FailingGenerator(GenerationError("The language model request failed: 503")),
    )
    result = pipeline.answer("firewall policy")
    assert "failed" in result["answer"].lower()


def test_rag_does_not_log_api_keys() -> None:
    assert "gsk_live_secret" not in redact_secrets("key=gsk_live_secret leftover")
    assert "[REDACTED]" in redact_secrets("gsk_live_secret")


def test_retrieval_hit_requires_expected_source() -> None:
    from evaluation.evaluate import retrieval_hit

    retrieved = [{"source": "sample_network_security.pdf", "page": 1}]
    assert retrieval_hit(retrieved, ["sample_network_security.pdf"]) is True
    assert retrieval_hit(retrieved, ["sample_document_processing.pdf"]) is False
    assert retrieval_hit(retrieved, []) is None


def test_evaluate_continues_after_api_failure(tmp_path: Path) -> None:
    from evaluation.evaluate import run_evaluation

    class StubPipeline:
        def answer(self, question: str) -> dict:
            if "timeout" in question:
                raise RuntimeError("simulated crash")
            if "fail-api" in question:
                return {
                    "answer": "The language model request failed: 503",
                    "sources": [],
                    "retrieved_chunks": [],
                    "retrieval_count": 0,
                    "latency": {
                        "retrieval_seconds": 0.1,
                        "generation_seconds": 0.0,
                        "total_seconds": 0.1,
                    },
                }
            return {
                "answer": "A firewall is only useful when its rules match a written policy.",
                "sources": [
                    {
                        "source": "sample_network_security.pdf",
                        "page": 1,
                        "chunk_id": "sample_network_security_p1_c01",
                        "score": 0.7,
                        "rank": 1,
                    }
                ],
                "retrieved_chunks": [],
                "retrieval_count": 1,
                "latency": {
                    "retrieval_seconds": 0.2,
                    "generation_seconds": 0.4,
                    "total_seconds": 0.6,
                },
            }

    dataset = tmp_path / "eval_dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "ok",
                        "question": "When is a firewall useful?",
                        "expected_answer": "When rules match a written policy.",
                        "expected_sources": ["sample_network_security.pdf"],
                    },
                    {
                        "id": "api",
                        "question": "fail-api please",
                        "expected_answer": "n/a",
                        "expected_sources": ["sample_network_security.pdf"],
                    },
                    {
                        "id": "crash",
                        "question": "timeout now",
                        "expected_answer": "n/a",
                        "expected_sources": ["sample_network_security.pdf"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "results.json"
    report_path = tmp_path / "report.json"
    payload = run_evaluation(
        pipeline=StubPipeline(),
        dataset_path=dataset,
        results_path=results_path,
        report_path=report_path,
    )
    assert len(payload["results"]) == 1
    assert len(payload["failures"]) == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["total_questions"] == 3
    assert report["completed"] == 1
    assert report["failed"] == 2
    assert report["retrieval_hit_rate"] == 1.0
    assert report["groundedness"] is None
    assert report["citation_score"] is None
    assert report["average_latency"] == 0.6


def _boom_client(error: Exception):
    class _Client:
        def __init__(self) -> None:
            self.chat = self
            self.completions = self

        def create(self, **kwargs):  # noqa: ANN003
            raise error

    return _Client()


def _sample_hit():
    from src.retrieve import RetrievedChunk

    return RetrievedChunk(
        rank=1,
        score=0.9,
        source="network_security.pdf",
        page=1,
        chunk_id="network_security_p1_c01",
        text="A firewall is only useful when its rules match a written policy.",
        vector_id=0,
    )


def test_generator_timeout_auth_and_rate_limit() -> None:
    import httpx
    from groq import APITimeoutError, AuthenticationError, RateLimitError

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    auth_response = httpx.Response(401, request=request)
    rate_response = httpx.Response(429, request=request)

    timeout_gen = Generator(client=_boom_client(APITimeoutError(request=request)))
    with pytest.raises(GenerationError) as timeout_info:
        timeout_gen.generate("q", [_sample_hit()])
    assert timeout_info.value.kind == "timeout"

    auth_gen = Generator(
        client=_boom_client(
            AuthenticationError("invalid key gsk_secret", response=auth_response, body=None)
        )
    )
    with pytest.raises(GenerationError) as auth_info:
        auth_gen.generate("q", [_sample_hit()])
    assert auth_info.value.kind == "auth"
    assert "gsk_secret" not in str(auth_info.value)

    rate_gen = Generator(
        client=_boom_client(RateLimitError("slow down", response=rate_response, body=None))
    )
    with pytest.raises(GenerationError) as rate_info:
        rate_gen.generate("q", [_sample_hit()])
    assert rate_info.value.kind == "rate_limit"


def test_rag_maps_groq_auth_and_rate_limit(tmp_path: Path) -> None:
    pipeline = _pipeline(
        tmp_path,
        _FailingGenerator(
            GenerationError("rejected", kind="auth"),
        ),
    )
    auth_result = pipeline.answer("firewall policy")
    assert "API key" in auth_result["answer"]
    assert auth_result["retrieval_count"] == 1

    pipeline = _pipeline(
        tmp_path,
        _FailingGenerator(GenerationError("limited", kind="rate_limit")),
    )
    rate_result = pipeline.answer("firewall policy")
    assert "rate-limited" in rate_result["answer"]


def test_invalid_groq_timeout_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.generate import validate_generation_env

    monkeypatch.setenv("GROQ_TIMEOUT_SECONDS", "not-a-number")
    with pytest.raises(GenerationError, match="GROQ_TIMEOUT_SECONDS"):
        validate_generation_env()


def test_rag_unexpected_exception_has_no_stack_in_answer(tmp_path: Path) -> None:
    class BoomRetriever:
        def search(self, question: str):  # noqa: ARG002
            raise RuntimeError("secret gsk_should_not_leak exploded")

    pipeline = RAGPipeline(retriever=BoomRetriever(), generator=_RecordingGenerator())
    result = pipeline.answer("firewall policy")
    assert result["answer"] == "An unexpected error occurred while answering."
    assert "gsk_should_not_leak" not in result["answer"]
    assert "Traceback" not in result["answer"]
