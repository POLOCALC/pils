"""Flight: load, synchronize, save and load the data of one flight.

The package splits :class:`Flight` by responsibility:

- :mod:`pils.flight.core`: :class:`Flight`, loading the raw data and the
  :mod:`pils.store` folder format (``save`` / ``load``)
- :mod:`pils.flight.sync`: ``Flight.sync``
- :mod:`pils.flight.containers`: :class:`RawData`, :class:`DroneData`,
  :class:`PayloadData`
"""

from pils.flight.containers import DroneData, PayloadData, RawData
from pils.flight.core import Flight

__all__ = ["DroneData", "Flight", "PayloadData", "RawData"]
