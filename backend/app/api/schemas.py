from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: str
    num_pages: int
    num_chunks: int

class QueryRequest(BaseModel):
    question: str = Field(min_length = 1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

class SourceChunk(BaseModel):
    chunk_id: str
    page: int
    text: str
    score: float

class QueryResponse(BaseModel):
    document_id: str
    question: str
    results: list[SourceChunk]

 
class CitationOut(BaseModel):
    ref: int
    chunk_id : str
    page: int
    text: str
    score: float

class AskResponse(BaseModel):
    document_id :str
    question: str
    answer: str
    grounded: bool
    citations: list[CitationOut]
