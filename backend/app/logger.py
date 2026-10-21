"""Structured logging setup.

Kept deliberately simple for now (stdlib logging with a consistent format).
Swap the formatter for a JSON formatter before production deployment so
logs are easy to query in Azure Monitor / Log Analytics.
"""

import logging
import sys
from pathlib import Path

from app.config.config import get_settings


def configure_logging(log_file: Path | None = None, *, console: bool = True) -> None:
    settings = get_settings()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)
    handlers: list[logging.Handler] = [handler] if console else []
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(handler.formatter)
        handlers.append(file_handler)

    root_logger.handlers = handlers





