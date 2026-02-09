"""Logging configuration helpers."""

from __future__ import annotations

import logging
import sys
from typing import Final

DEFAULT_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format=DEFAULT_FORMAT,
        stream=sys.stdout,
    )

    logging.getLogger("uvicorn.error").setLevel(level.upper())
    logging.getLogger("uvicorn.access").setLevel(level.upper())
