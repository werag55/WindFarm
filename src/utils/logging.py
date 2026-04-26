"""Logging configuration for the offshore wind project."""

import logging

def configure_logging(level: int = logging.INFO) -> None:
    """Configure a simple shared logging format once."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
