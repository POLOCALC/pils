"""Lab measurement folders: notes (YAML front matter) and file discovery.

A lab measurement lives in ``<item>/<test_type>/`` as files sharing a stem::

    20251101_140000.md          # notes: YAML front matter + description
    20251101_140000_raw.h5py    # raw data (.h5py, .s2p, ...)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*(\n|$)", re.DOTALL)
RAW_SUFFIX = "_raw"


def load_notes(path: str | Path) -> dict[str, Any]:
    """The ``config`` block of a measurement notes file.

    Raises
    ------
    ValueError
        If the file has no YAML front matter or no ``config`` key.
    """
    content = Path(path).read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(content)
    if not match:
        raise ValueError(f"No configuration block (--- ... ---) in {path}")
    front_matter = yaml.safe_load(match.group(1)) or {}
    if "config" not in front_matter:
        raise ValueError(f"No 'config' key in the front matter of {path}")
    return front_matter["config"]


def component_ids(config: dict[str, Any]) -> dict[str, str]:
    """STOUT item ids in a notes config: role -> item_id.

    Main components keep their role (``multiplier``, ``detector``,
    ``receiving_antenna``...); auxiliary ones are prefixed with ``aux.``,
    nested groups with their key (``aux.waveguides.STR``).
    """
    ids: dict[str, str] = {}

    def walk(prefix: str, node: Any) -> None:
        if not isinstance(node, dict):
            return
        if "item_id" in node:
            ids[prefix] = str(node["item_id"])
            return
        for key, value in node.items():
            walk(f"{prefix}.{key}" if prefix else str(key), value)

    walk("", config.get("components", {}))
    walk("aux", config.get("auxiliary_components", {}))
    return ids


@dataclass
class LabMeasurement:
    """One measurement in a test folder: its stem, raw file and notes.

    Parameters
    ----------
    folder : Path
        The ``<item>/<test_type>`` folder.
    stem : str
        Shared file name prefix, usually ``YYYYMMDD_HHMMSS``.
    """

    folder: Path
    stem: str
    raw_path: Path | None = None
    notes_path: Path | None = None
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def test_type(self) -> str:
        return self.folder.name

    @classmethod
    def find(cls, folder: str | Path, stem: str | None = None) -> "LabMeasurement":
        """The measurement ``stem`` in ``folder`` (latest one if None).

        Raises
        ------
        FileNotFoundError
            If the folder has no raw file (with that stem).
        """
        folder = Path(folder)
        stems = list_measurements(folder)
        if not stems:
            raise FileNotFoundError(f"No raw measurement (*{RAW_SUFFIX}.*) in {folder}")
        if stem is None:
            stem = stems[-1]
        elif stem not in stems:
            raise FileNotFoundError(f"No raw measurement '{stem}' in {folder}")
        raw = sorted(folder.glob(f"{stem}{RAW_SUFFIX}.*"))[0]
        notes = folder / f"{stem}.md"
        return cls(
            folder=folder,
            stem=stem,
            raw_path=raw,
            notes_path=notes if notes.is_file() else None,
            config=load_notes(notes) if notes.is_file() else {},
        )


def list_measurements(folder: str | Path) -> list[str]:
    """Stems of the raw measurements in a test folder, oldest first."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    stems = {
        p.name.split(RAW_SUFFIX + ".")[0]
        for p in folder.iterdir()
        if p.is_file() and RAW_SUFFIX + "." in p.name
    }
    return sorted(stems)
