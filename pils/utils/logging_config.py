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


def setup_logging(
    level: str = "INFO", log_file: Path | None = None, console_output: bool = True
) -> None:
    """
    Configure logging for PILS package.

    Parameters
    ----------
    level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : Optional[Path], optional
        Optional path to log file (default: logs/pils.log).
    console_output : bool, optional
        Enable console output handler.

    Examples
    --------
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
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module).

    Returns
    -------
    logging.Logger
        Logger instance configured for PILS.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing flight data")
    """
    return logging.getLogger(name)
