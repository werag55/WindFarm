"""Logging configuration for the offshore wind project."""

import logging

def configure_logging() -> None:
    """Configure a simple shared logging format once."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
