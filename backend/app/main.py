import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config.config import get_settings
from app.logger import configure_logging

settings = get_settings()
configure_logging()
logger = logging.getLogger(__name__)

#  It turns a standard asynchronous function into a structured context manager.
@asynccontextmanager
async def lifespan(_:FastAPI):
    logger.info("%s starting up in '%s' environment.", settings.app_name, settings.environment)
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

