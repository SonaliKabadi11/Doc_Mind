from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile

from app.api.schemas import (
    AskResponse,
    CitationOut,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from app.config.config import Settings, get_settings
from app.services.rag_service import RAGService
from app.models.models import Answer

router = APIRouter()

def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service

RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]



@router.get("/health")
def health_check() -> dict[str, str]:
    """Liveness/readiness probe target for container orchestration."""
    settings = get_settings()
    return {"status": "working", "app_name": settings.app_name, "environment": settings.environment}

# Plain `def` (not async): embedding is CPU-bound, so FastAPI runs these in a
# threadpool instead of blocking the event loop.
@router.post("/documents", response_model=IngestResponse, status_code=201)
def upload_document(file: UploadFile, service: RAGServiceDep, settings: SettingsDep):
    if file.content_type != "application/pdf":
        raise HTTPException(415, "Only PDF uploads are supported")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")

    result = service.ingest(BytesIO(data))
    return IngestResponse(
        document_id = result.document_id,
        num_pages = result.num_pages,
        num_chunks = result.num_chunks
    )

@router.post("/documents/{document_id}/query", response_model = QueryResponse)
def query_document(document_id: str, body: QueryRequest, service: RAGServiceDep):
    result = service.retrieve(document_id, body.question, body.top_k)
    return QueryResponse(
        document_id = document_id,
        question = body.question,
        results = [SourceChunk(
            chunk_id=r.chunk.chunk_id,
            page=r.chunk.page,
            text=r.chunk.text,
            score = round(r.score, 4)
        ) for r in result]
    )

@router.post("/documents/{document_id}/ask", response_model=AskResponse)
def ask_document(document_id:str, body:QueryRequest, service:RAGServiceDep):
    answer = service.answer(document_id, body.question, body.top_k)
    return _to_ask_response(document_id, body.question, answer)

def _to_ask_response(document_id:str, question: str, answer:Answer) -> AskResponse:
    return AskResponse(
        document_id=document_id,
        question=question,
        answer=answer.text,
        grounded=answer.grounded,
        citations=[
           CitationOut(ref=c.ref, chunk_id=c.chunk_id, page=c.page, text=c.text, score=round(c.score, 4))
            for c in answer.citations
        ]
    )

@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, service: RAGServiceDep) -> Response:
    service.delete(document_id)
    return Response(status_code=204)

