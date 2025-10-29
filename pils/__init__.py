"""
PILS - Payload Instrument Loading System

Tools to load and visualize data from drone missions, including drone logs,
payload sensor data, and Litchi flight logs.
"""

__version__ = "0.1.0"

# Main handler classes
from pils.pathhandler import PathHandler
from pils.datahandler import DataHandler, Payload, Drone

# Report generation (optional dependencies)
try:
    from pils.report import FlightReport, quick_report
    _REPORT_AVAILABLE = True
except ImportError:
    _REPORT_AVAILABLE = False
    FlightReport = None
    quick_report = None

# Tools and utilities
from pils.tools import *


def enable_stout_logging(verbose: bool = True):
    """
    Enable stout logging messages.

    By default, pils runs in silent mode to avoid cluttering console output.
    Call this function if you want to see stout's internal logging messages.

    Args:
        verbose: If True, enables logging. If False, disables it (silent mode).

    Example:
        >>> import pils
        >>> pils.enable_stout_logging()  # Enable logging
        >>> handler = pils.PathHandler("flight_20240715_1430")
    """
    try:
        from stout.services import set_silent_mode

        set_silent_mode(not verbose)
    except ImportError:
        pass  # stout not available, ignore


__all__ = [
    "PathHandler",
    "DataHandler",
    "Payload",
    "Drone",
    "FlightReport",
    "quick_report",
    "enable_stout_logging",
    "__version__",
]
