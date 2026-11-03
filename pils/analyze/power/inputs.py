"""The lab results a flight's power analysis needs, checked before it runs.

For each component of the payload's RF chain (coupler, antenna, attenuators,
detector) the measurement the analysis would read at the flight's take-off:
found, made only after the flight, or missing (and why).
"""

from datetime import datetime
from typing import Any

from pils.analyze.power.rf_chain import payload_items
from pils.lab_analysis.store import Item, LabDataNotFound, LabStore, value_at

# (category, test type, what it gives, the lookup), as the power analysis does them
NEEDS: list[tuple[str, str, str, Any]] = [
    (
        "coupler",
        "transmission",
        "coupler transmission",
        lambda s, i, f, _: s.transmission(i, "transmission", f),
    ),
    (
        "coupler",
        "coupling_performance",
        "coupled-port coupling",
        lambda s, i, f, _: s.transmission(i, "coupling_performance", f),
    ),
    (
        "antenna",
        "radiation_pattern",
        "antenna gain",
        lambda s, i, f, _: value_at(s.antenna_gain(i), f, "gain_dbi", "gain_err_db"),
    ),
    (
        "antenna",
        "radiation_pattern",
        "antenna beam width",
        lambda s, i, f, _: s.antenna_fwhm(i),
    ),
    (
        "antenna",
        "insertion_loss",
        "antenna insertion loss",
        lambda s, i, f, _: s.insertion_loss(i, f),
    ),
    (
        "antenna",
        "impedance_matching",
        "antenna reflection",
        lambda s, i, f, d: s.reflection(i, f, dataset=d),
    ),
    (
        "attenuator",
        "attenuation",
        "attenuation",
        lambda s, i, f, _: s.transmission(i, "attenuation", f),
    ),
    (
        "detector",
        "responsivity",
        "diode responsivity",
        lambda s, i, f, _: s.responsivity(i),
    ),
]
# Optional in a payload: none of them is fine
OPTIONAL = {"attenuator"}


def check(
    payload: dict[str, Any],
    frequency_ghz: float | None,
    store: LabStore,
    as_of: datetime | None = None,
    impedance_dataset: str = "fsp_impedance_kapton",
) -> list[dict[str, Any]]:
    """One row per lab result the power analysis reads.

    Each row: ``category``, ``item`` (name, or None when the payload has none),
    ``test_type``, ``quantity``, ``status`` (``ok``, ``after`` -- made only
    after ``as_of`` --, ``missing`` or ``no item``), ``detail``, and the
    ``measurement``, ``measured_at``, ``source`` and ``path`` read.
    """
    store.as_of = as_of
    rows = []
    for category, test_type, quantity, lookup in NEEDS:
        try:
            items: list[Item] = payload_items(payload, category)
        except ValueError:
            if category not in OPTIONAL:
                rows.append(
                    {
                        "category": category,
                        "item": None,
                        "test_type": test_type,
                        "quantity": quantity,
                        "status": "no item",
                        "detail": f"No {category} in the payload",
                    }
                )
            continue
        for item in items:
            row: dict[str, Any] = {
                "category": category,
                "item": item.get("name"),
                "test_type": test_type,
                "quantity": quantity,
            }
            before = len(store.used)
            if frequency_ghz is None and quantity not in (
                "antenna beam width",
                "diode responsivity",
            ):
                row.update(status="missing", detail="The flight has no frequency")
                rows.append(row)
                continue
            try:
                lookup(store, item, frequency_ghz, impedance_dataset)
                status, detail = "ok", ""
            except (LabDataNotFound, FileNotFoundError, KeyError, ValueError) as e:
                status, detail = "missing", str(e).splitlines()[0][:200]
            read = store.used[before:] or [
                u
                for u in store.used
                if u["item"] == item.get("name") and u["test_type"] == test_type
            ]
            if read:
                used = read[-1]
                row.update(
                    measurement=used["measurement"],
                    measured_at=used["measured_at"],
                    source=used["source"],
                    path=used["path"],
                )
                if status == "ok" and used["after"]:
                    status = "after"
                    detail = "Measured only after the flight"
            row.update(status=status, detail=detail)
            rows.append(row)
    return rows
