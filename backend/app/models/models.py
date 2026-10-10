from dataclasses import dataclass


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


