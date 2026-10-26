"""
PILS Lab Analysis Modules.

Versioned analyses of component measurements made in the lab (VNA,
antenna beam patterns, multiplier output power and diode responsivity),
and access to the lab results of inventory items.
"""

import importlib
from typing import Any

# Imported when first used: the lab store needs none of the analyses' dependencies
_LAZY = {
    "LabAnalysis": "pils.lab_analysis.base",
    "LabVersion": "pils.lab_analysis.base",
    "BeamAnalysis": "pils.lab_analysis.beam",
    "MultiplierDiodeAnalysis": "pils.lab_analysis.multiplier_diode",
    "LabMeasurement": "pils.lab_analysis.notes",
    "load_notes": "pils.lab_analysis.notes",
    "LabDataNotFound": "pils.lab_analysis.store",
    "LabStore": "pils.lab_analysis.store",
    "VNAAnalysis": "pils.lab_analysis.vna",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module 'pils.lab_analysis' has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


__all__ = [
    "BeamAnalysis",
    "LabAnalysis",
    "LabDataNotFound",
    "LabMeasurement",
    "LabStore",
    "LabVersion",
    "MultiplierDiodeAnalysis",
    "VNAAnalysis",
    "load_notes",
]
