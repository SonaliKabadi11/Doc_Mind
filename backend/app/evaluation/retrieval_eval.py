"""Retrieval evaluation: recall@k, MRR, evidence ceiling and gate scores. Makes NO LLM calls.

Relevance is defined by evidence quotes, not chunk ids or page numbers, so the same dataset stays valid when chunk size, overlap or the embedding model change.
A chunk is relevant if it fully contains ANY of the question's evidence quotes.
"""

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np

from app.evaluation.dataset import Question
from app.evaluation.text import best_status, locate, squash
from app.models import Chunk, Config, Experiment, Index, QuestionResult, Summary
from app.retrieval.in_memory_store import InMemoryStore

MAX_K = 20
KS = (1, 3, 5, 8, 10, 20)
GATE_MARGIN = 0.03

class Ingestor(Protocol):
    def extract_pages(self, source: str | Path | BinaryIO) -> list[tuple[int, str]]: ...
    def chunk_pages(self, document_id: str, pages: list[tuple[int, str]]) -> list[Chunk]: ...
    def embed_chunks(self, chunks: list[str]) -> np.ndarray: ...
    def embed_query(self, query: str) -> np.ndarray: ...

class CachingIngestor:
    """Wraps an ingestor and caches chunk embeddings on disk.

    The key is the model name plus the exact chunk texts, so a changed splitter or model can
    never serve stale vectors, while repeated runs (changing only metrics or k) are instant.
    """
    def __init__(self, inner:Ingestor, model_name:str, cache_dir:Path |None) ->None:
        self.inner, self.model_name, self.cache_dir = inner, model_name, cache_dir
        self.hits = self.misses = 0
    def extract_pages(self, source):
        return self.inner.extract_pages(source)

    def chunk_pages(self, document_id, pages):
        return self.inner.chunk_pages(document_id, pages)

    def embed_query(self, query):
        return self.inner.embed_query(query)

    def embed_chunks(self, chunks):
        if self.cache_dir is None:
            return self.inner.embed_chunks(chunks)
        key = hashlib.sha256((self.model_name + "\x00" + "\x00".join(chunks)).encode()).hexdigest()
        path = self.cache_dir/f"{key}.npy"
        if path.exists():
            self.hits+=1
            return np.load(path)
        self.misses += 1
        embeddings = self.inner.embed_chunks(chunks)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        return embeddings

def build_index(ingestor:Ingestor, doc:str, pdf:Path) -> Index:
    pages = ingestor.extract_pages(pdf)
    chunks = ingestor.chunk_pages(doc,pages)
    if not chunks:
        raise ValueError(f"{doc}: no extractable text in {pdf}")
    embeddings = ingestor.embed_chunks([c.text for c in chunks])
    store = InMemoryStore()
    store.add_chunks(doc, chunks, embeddings)
    return Index(
        doc=doc,
        chunks = chunks,
        squashed_chunks = [squash(c.text) for c in chunks],
        squashed_doc = squash("".join(text for _, text in pages)),
        store = store,
    )

def evidence_status(evidence : Iterable[str], index:Index)-> str:
    return best_status([locate(e,index.squashed_chunks, index.squased_doc) for e in evidence])

def evaluate_index(index:Index, ingestor:Ingestor, questions:list[Question]) -> list[QuestionResult]:
    rows = []
    for q in questions:
        results = index.store.search(index.doc, ingestor.embed_query(q.question), MAX_K)
        top = results[0].score if results else 0.0
        if not q.answerable:
            rows.append(QuestionResult(q.id, q.doc, q.type, False, top, "n/a", None))
            continue
        needles = [n for n in map(squash, q.evidence) if n]
        rank = next(
            (i for i, r in enumerate(results, 1) if any(n in squash(r.chunk.text) for n in needles)),
            None,
        )
        status = best_status([locate(e, index.squashed_chunks, index.squashed_doc) for e in q.evidence])
        rows.append(QuestionResult(q.id, q.doc, q.type, True, top, status, rank))
    return rows


def summarize(results: Iterable[QuestionResult], ks: tuple[int, ...] = KS) -> Summary:
    answerable = [r for r in results if r.answerable]
    hits = {k: sum(r.rank is not None and r.rank <= k for r in answerable) for k in ks}
    mrr = sum(1/r.rank for r in answerable if r.rank) / len(answerable) if answerable else 0.0
    return Summary(
        len(answerable),
        hits,
        mrr, 
        sum(r.evidence_status == "intact" for r in answerable)
    )

def run_experiment(config: Config, make_ingestor: Callable, docs: dict[str,Path], questions: list[Question]) -> Experiment:
    ingestor = make_ingestor(config)
    by_doc: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_doc[q.doc].append(q)
    results: list[QuestionResult] = []

    num_chunks = 0
    for doc, doc_questions in by_doc.items():
        index = build_index(ingestor, doc, docs[doc])
        num_chunks += len(index.chunks)
        results += evaluate_index(index, ingestor, doc_questions)
    return Experiment(config, results, num_chunks)

# ---------------------------------------------------------------- reporting

def _frac(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.0%})" if total else "n/a"

def format_table(experiments: list[Experiment]) -> str:
    n = summarize(experiments[0].results).n
    lines = [
        f"RETRIEVAL ({n} answerable questions). Hit@k = a relevant chunk is in the top k; "
        "intact = evidence not cut by chunking",
        f"{'size/overlap':<14}{'chunks':>7}{'intact':>13}" + "".join(f"{f'Hit@{k}':>13}" for k in KS) + f"{'MRR':>7}",
    ]
    for exp in experiments:
        s = summarize(exp.results)
        print(s)
        lines.append(
            f"{exp.config.label:<14}{exp.num_chunks:>7}{_frac(s.intact, s.n):>13}"
            + "".join(f"{_frac(s.hits[k], s.n):>13}" for k in KS)
            + f"{s.mrr:>7.3f}"
        )
    return "\n".join(lines)

def format_breakdown(exp: Experiment) -> str:
    """Hit rate and MRR per document and per question type."""
    lines = [f"=== {exp.config.label} ==="]
    for title, key in (("by document", lambda r: r.doc), ("by question type", lambda r: r.type)):
        groups: dict[str, list[QuestionResult]] = defaultdict(list)
        for r in exp.results:
            if r.answerable:
                groups[key(r)].append(r)
        lines.append(title + ":")
        for name, rows in sorted(groups.items()):
            s = summarize(rows)
            lines.append(f"  {name:<16} Hit@5={_frac(s.hits[5], s.n):<12} Hit@10={_frac(s.hits[10], s.n):<12} MRR={s.mrr:.3f}")
    return "\n".join(lines)

def format_misses(exp: Experiment, k: int = 5) -> str:
    """Answerable questions with no relevant chunk in the top k: where to look first."""
    lines = [f"misses at k={k} ({exp.config.label}):"]
    for r in exp.results:
        if r.answerable and (r.rank is None or r.rank > k):
            lines.append(f"  {r.id:<18} {r.type:<8} evidence={r.evidence_status:<8} rank={r.rank or '>' + str(MAX_K)}")
    return "\n".join(lines) if len(lines) > 1 else lines[0] + " none"