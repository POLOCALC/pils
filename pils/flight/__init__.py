"""Flight: load, synchronize, save and load the data of one flight.

The package splits :class:`Flight` by responsibility:

- :mod:`pils.flight.core`: :class:`Flight`, loading the raw data and the
  :mod:`pils.store` folder format (``save`` / ``load``)
- :mod:`pils.flight.sync`: ``Flight.sync``
- :mod:`pils.flight.hdf5`: the older single-HDF5 format (``to_hdf5`` /
  ``from_hdf5``)
- :mod:`pils.flight.containers`: :class:`RawData`, :class:`DroneData`,
  :class:`PayloadData`

Every name of the former ``pils/flight.py`` module is re-exported here.
"""

# Names the former module imported, which callers may still look up here
from pils.drones.BlackSquareDrone import BlackSquareDrone  # noqa: F401
from pils.drones.DJIDrone import DJIDrone  # noqa: F401
from pils.drones.litchi import Litchi  # noqa: F401
from pils.flight.containers import DroneData, PayloadData, RawData
from pils.flight.core import (  # noqa: F401
    Flight,
    _attribute_alias,
    _standardize,
    _standardize_records,
)
from pils.flight.hdf5 import (  # noqa: F401
    _LEGACY_SYNC_COORDINATES,
    _SYNC_KINDS,
    _deserialize_from_hdf5,
    _get_current_timestamp,
    _get_package_version,
    _legacy_sync_table,
    _parse_temporal_dtype,
    _serialize_for_hdf5,
)
from pils.flight.sync import (  # noqa: F401
    MAX_GPS_H_ACC_M,
    MIN_REFERENCE_COVERAGE,
    _time_coverage,
    _valid_positions,
)
from pils.sensors.camera import Camera  # noqa: F401
from pils.sensors.sensors import sensor_config  # noqa: F401
from pils.synchronizer import Synchronizer  # noqa: F401

__all__ = ["DroneData", "Flight", "PayloadData", "RawData"]
