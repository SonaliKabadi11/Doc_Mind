"""Structured logging setup.

Kept deliberately simple for now (stdlib logging with a consistent format).
Swap the formatter for a JSON formatter before production deployment so
logs are easy to query in Azure Monitor / Log Analytics.
"""

import logging
import sys

from app.config.config import get_settings


def configure_logging() -> None:
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
    root_logger.handlers = [handler]





