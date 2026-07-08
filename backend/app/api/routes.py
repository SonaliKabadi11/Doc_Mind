from fastapi import APIRouter

from app.config.config import get_settings

router = APIRouter()

@router.get("/health")
def health_check() -> dict[str, str]:
    """Liveness/readiness probe target for container orchestration."""
    settings = get_settings()
    return {"status": "working" , "app_name": settings.app_name, "environment": settings.environment}

