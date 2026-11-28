"""The telemetry LAGER logs in a flight's data folder, decoded as its YAML says.

The folder holds the configuration the run used (``*.yaml``, with the
``uav`` and ``gimbal`` blocks, at the top or under ``pointing_controller``)
and the logs it names: ``uav_telemetry.bin`` and ``gimbal_telemetry.bin``.
The ``name`` of each block picks the decoder, so a flight of a Matrice 600
and one of a Matrice 400 are read by the same call::

    telemetry = load_telemetry(folder)
    telemetry.uav.data       # polars table of the drone's telemetry
    telemetry.gimbal.data    # polars table of the gimbal's
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..gimbals.gremsy_t7 import GremsyT7Telemetry
from ..utils.logging_config import get_logger
from .dji.m400.telemetry import M400Telemetry
from .dji.m600.telemetry import M600Telemetry

logger = get_logger(__name__)

# Decoder of each device, by its name in the configuration ("DJI M600" -> "djim600")
UAV_DECODERS = {"djim600": M600Telemetry, "djim400": M400Telemetry}
GIMBAL_DECODERS = {"gremsyt7": GremsyT7Telemetry}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


@dataclass
class Telemetry:
    """The decoded logs of a data folder; a device that was not configured or has no log is None."""

    config: dict[str, Any]
    config_file: Path
    uav: M600Telemetry | M400Telemetry | None = None
    gimbal: GremsyT7Telemetry | None = None
    uav_model: str | None = None
    gimbal_model: str | None = None
    missing: list[str] = field(default_factory=list)


def read_config(folder: str | Path) -> tuple[dict[str, Any], Path]:
    """Read the device blocks (``uav``, ``gimbal``, ...) of the folder's YAML, and its path.

    Several YAML files: the first (by name) that has a ``uav`` or a ``gimbal``
    block. Raises ``FileNotFoundError`` if there is none.
    """
    folder = Path(folder)
    for file in sorted([*folder.glob("*.yaml"), *folder.glob("*.yml")]):
        try:
            content = yaml.safe_load(file.read_text())
        except (yaml.YAMLError, OSError) as e:
            logger.warning("%s: not readable as YAML: %s", file, e)
            continue
        if not isinstance(content, dict):
            continue
        block = content.get("pointing_controller", content)
        if isinstance(block, dict) and ("uav" in block or "gimbal" in block):
            return block, file
    raise FileNotFoundError(
        f"No YAML configuration with a 'uav' or 'gimbal' block in {folder}"
    )


def _decoder(table: dict, block: Any, kind: str):
    name = block.get("name") if isinstance(block, dict) else None
    if name is None:
        raise ValueError(f"The '{kind}' block of the configuration has no 'name'")
    if _key(name) not in table:
        raise ValueError(f"No decoder for {kind} {name!r}; known: {sorted(table)}")
    return table[_key(name)]


def load_telemetry(folder: str | Path) -> Telemetry:
    """Decode the drone and gimbal logs of a data folder, the decoders chosen by its YAML.

    A configured device whose log is not in the folder is left None and named in
    ``missing``; an unknown device name raises ``ValueError``.
    """
    folder = Path(folder)
    config, config_file = read_config(folder)
    result = Telemetry(config=config, config_file=config_file)
    for kind, table in (("uav", UAV_DECODERS), ("gimbal", GIMBAL_DECODERS)):
        if kind not in config:
            continue
        decoder = _decoder(table, config[kind], kind)
        setattr(result, f"{kind}_model", str(config[kind]["name"]))
        loader = decoder(folder)
        try:
            loader.load_data()
        except FileNotFoundError as e:
            logger.warning("%s", e)
            result.missing.append(kind)
            continue
        setattr(result, kind, loader)
    return result
