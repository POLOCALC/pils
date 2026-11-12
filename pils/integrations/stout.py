"""STOUT, the POLOCALC campaign and inventory database: the one way pils reads it.

This module is the only place in pils that does ``import stout`` (lazily,
in :func:`_module`); everything else asks it here.

* :data:`layout` is ``stout.data_layout``: the data folder's layout (folder
  names, flight_/plan_/rev_ names in UTC, the flights in campaigns/ and
  field_tests/). STOUT owns it; pils analyses what is in it and never spells
  it itself (``from pils.integrations.stout import layout``).
* The other functions read STOUT's database: each returns data, or raises
  :class:`StoutUnavailable` when STOUT is not installed
  (:func:`is_available` tells which before asking).

STOUT is a dependency of pils (from POLOCALC's repository); ``pils[stout]``
is kept as an alias.

Examples
--------
>>> import pils.integrations.stout as stout_db
>>> if stout_db.is_available():
...     flights = stout_db.campaign_flights("202511")
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

__all__ = [
    "StoutNotFound",
    "StoutUnavailable",
    "all_campaigns",
    "all_flights",
    "campaign_flights",
    "data_path",
    "drone_model",
    "flight",
    "inventory_item",
    "is_available",
    "payload",
]

INSTALL_HINT = "pip install 'stout @ git+https://github.com/POLOCALC/stout.git@main'"

# One service of each class (a STOUT service opens the database)
_services: dict[Any, Any] = {}


class StoutUnavailable(ImportError):
    """STOUT is not installed (an ImportError, for callers that catch those)."""


class StoutNotFound(LookupError):
    """STOUT has no such record."""


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise StoutUnavailable(f"STOUT is not installed ({INSTALL_HINT}): {e}") from e


def __getattr__(name: str) -> Any:
    # ``layout``: stout.data_layout, imported on first use
    if name == "layout":
        return _module("stout.data_layout")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _service(module: str, name: str) -> Any:
    cls = getattr(_module(module), name)
    if cls not in _services:
        _services[cls] = cls()
    return _services[cls]


def _campaigns() -> Any:
    return _service("stout.services.campaigns", "CampaignService")


def _inventory() -> Any:
    return _service("stout.services.inventory.service", "InventoryService")


def is_available() -> bool:
    """Whether STOUT can be imported."""
    try:
        _module("stout.config")
    except StoutUnavailable:
        return False
    return True


def data_path() -> Path:
    """STOUT's data folder (``Config.data_root()``: ``STOUT_DATA_PATH`` read now)."""
    return Path(_module("stout.config").Config.data_root())


def all_campaigns() -> list[dict[str, Any]]:
    """Return every campaign in the database."""
    return _campaigns().get_all_campaigns()


def all_flights() -> list[dict[str, Any]]:
    """Return every flight of every campaign, by take-off."""
    return _campaigns().get_all_flights()


def campaign_flights(campaign_id: str) -> list[dict[str, Any]]:
    """Return the flights of a campaign (by id)."""
    return _campaigns().get_flights_by_campaign(campaign_id=campaign_id)


def flight(
    flight_id: str | None = None, flight_name: str | None = None
) -> dict[str, Any] | None:
    """Return one flight, by id or name, or None."""
    return _campaigns().get_flight(flight_name=flight_name, flight_id=flight_id)


def payload(payload_id: str) -> dict[str, Any] | None:
    """Return a payload with its items, or None."""
    return _campaigns().get_payload_by_id(payload_id)


def inventory_item(item_id: str) -> dict[str, Any]:
    """Return an inventory item.

    Raises
    ------
    StoutNotFound
        If the inventory has no such item.
    StoutUnavailable
        If STOUT is not installed.
    """
    not_found = _module("stout.exceptions").NotFoundError
    try:
        item = _inventory().get_item_by_id(item_id)
    except not_found as e:
        raise StoutNotFound(f"No inventory item {item_id}") from e
    if not item:
        raise StoutNotFound(f"No inventory item {item_id}")
    return item


def drone_model(drone_id: str) -> str | None:
    """Return a drone's model as the inventory names it (lower case), or None.

    Its ``specifications.model``, else its name, else its category.
    """
    item = inventory_item(drone_id)
    specs = item.get("specifications") if isinstance(item, dict) else None
    specs = specs if isinstance(specs, dict) else {}
    model = specs.get("model") or item.get("name") or item.get("category")
    return model.lower() if isinstance(model, str) else None
