from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.in_memory_store import InMemoryStore

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    page: int  # 1-based
    index: int  # position within the document


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
   

@dataclass(frozen=True)
class Citation:
    ref: int  # the [n] marker used in the answer text
    chunk_id : str
    page: int
    text: str
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    grounded: bool
    citations: list[Citation]
    retrieved: list[SearchResult]  # everything that went into the prompt

@dataclass
class Prompt:
    system: str
    user: str
    sources: list[SearchResult] # sources[i] is cited in the answer as [i + 1]

@dataclass 
class ParsedAnswer:
    text: str # answer with invalid markers removed
    citations: list[Citation] # valid refs, in order of first appearance
    invalid_refs: list[int] # refs the model cited that don't exist (for logging)


@dataclass(frozen=True)
class Question:
    id: str
    doc: str
    question: str
    answerable: bool
    type: str
    split: str
    evidence: tuple[str, ...] = ()
    refernce_answer: str | None = None

@dataclass
class Index:
    doc: str
    chunks: list[Chunk]
    squashed_chunks: list[str]
    squashed_doc: str
    store: "InMemoryStore"

@dataclass(frozen=True)
class QuestionResult:
    id: str
    doc: str
    type: str
    answerable: bool
    top_score: float  # similarity of the best-ranked chunk (relevant or not)
    evidence_status: str  # intact | split | missing | n/a
    rank: int | None  # 1-based rank of first chunk containing evidence (within MAX_K)


@dataclass(frozen=True)
class Summary:
    n: int  # answerable questions
    hits: dict[int, int]  # k -> number of questions with a relevant chunk in the top k
    mrr: float
    intact: int  # questions whose evidence survives chunking (the recall ceiling)

@dataclass(frozen=True)
class Config:
    model: str
    chunk_size: int
    chunk_overlap: int

    @property
    def label(self) -> str:
        return f"{self.chunk_size}/{self.chunk_overlap}"

@dataclass
class Experiment:
    config: Config
    results: list[QuestionResult]
    num_chunks: int