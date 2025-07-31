"""
Logger configuration for the application.
"""
import logging
import sys
from rich.logging import RichHandler

def setup_logger(name: str) -> logging.Logger:
    """Sets up a standardized logger."""
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[])]
    )
    return logging.getLogger(name)

def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance."""
    return logging.getLogger(name)
