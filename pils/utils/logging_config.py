"""
Centralized logging configuration for PILS.

This module provides a unified logging setup for the entire PILS package.
Usage:
    from pils.utils.logging_config import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO", log_file: Optional[Path] = None, console_output: bool = True
) -> None:
    """
    Configure logging for PILS package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file (default: logs/pils.log)
        console_output: Enable console output handler

    Example:
        >>> setup_logging(level="DEBUG", log_file=Path("my_logs/pils.log"))
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger("pils")
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatter with timestamp, level, module, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler with colors (if supported)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Logger instance configured for PILS

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing flight data")
    """
    return logging.getLogger(name)
