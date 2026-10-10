import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config.config import get_settings
from app.exception import DocumentNotFoundError, IngestionError,  LLMError, LLMTimeoutError
from app.logger import configure_logging

settings = get_settings()
configure_logging()
logger = logging.getLogger(__name__)

#  It turns a standard asynchronous function into a structured context manager.
@asynccontextmanager
async def lifespan(app:FastAPI):
    from app.ingestion.base import Ingestion
    from app.retrieval.in_memory_store import InMemoryStore
    from app.services.rag_service import RAGService
    from app.generation.factory import build_llm_provider


    logger.info("%s starting up in '%s' environment.", settings.app_name, settings.environment)
    llm = build_llm_provider(settings)
    ingestion = Ingestion(
        model_name = settings.embedding_model,
        chunk_size = settings.chunk_size,
        chunk_overlap = settings.chunk_overlap
    )
    app.state.rag_service = RAGService(ingestion, 
                                       InMemoryStore(),
                                       llm,
                                       similarity_threshold=settings.similarity_threshold,
                                       max_context_chars=settings.max_context_chars)
    yield
    logger.info("%s shutting down.", settings.app_name)



app = FastAPI(
    title=settings.app_name,
    description="Retrieval-augmented Q&A over uploaded documents.",
    version="0.1.0",
    lifespan = lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)




app.include_router(router)
@app.exception_handler(DocumentNotFoundError)
async def not_found_handler(_: Request, exc: DocumentNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(IngestionError)
async def ingestion_handler(_: Request, exc: IngestionError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})

# Fixed messages on purpose: provider details belong in the logs, not in API responses.
@app.exception_handler(LLMTimeoutError)
async def ll_timeout_handler(_:Request, exc: LLMTimeoutError) -> JSONResponse:
    return JSONResponse(
        status_code=504, 
        content={
            "detail": "The language model took too long to respond"
        }
    )

@app.exception_handler(LLMError)
async def llm_error_handler(_:Request, exc: LLMTimeoutError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={
            "detail": "The language model is unavailable.Try again later"
        }
    )