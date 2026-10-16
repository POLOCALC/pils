"""Source RF chain: from the diode reading to the radiated power.

The payload source is multiplier -> directional coupler -> attenuators ->
antenna, with the diode detector on the coupled port. Each component's lab
results come from :class:`pils.lab_analysis.LabStore`.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import constants

from pils.lab_analysis.store import Item, LabStore, value_at
from pils.utils.units import db_to_linear, linear_to_db, mw_to_dbm


def payload_items(payload: dict[str, Any], category: str) -> list[Item]:
    """Items of a category in a payload (STOUT payload dict)."""
    items = [i for i in payload.get("items", []) if i.get("category") == category]
    if not items:
        raise ValueError(f"No {category} in payload {payload.get('name', '')}")
    return items


def payload_item(payload: dict[str, Any], category: str) -> Item:
    """The single item of a category in a payload."""
    items = payload_items(payload, category)
    if len(items) > 1:
        names = [i.get("name") for i in items]
        raise ValueError(f"Several {category} items in the payload: {names}")
    return items[0]


@dataclass
class RFChain:
    """Lab-measured RF chain of a payload at one frequency (dB values)."""

    frequency_ghz: float
    coupler: str
    coupler_transmission_db: float
    coupler_transmission_err_db: float
    coupler_coupling_db: float
    coupler_coupling_err_db: float
    antenna: str
    antenna_gain_dbi: float
    antenna_gain_err_db: float
    antenna_insertion_loss_db: float
    antenna_reflection_db: float
    antenna_reflection_err_db: float
    attenuators: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        frequency_ghz: float,
        store: LabStore,
        impedance_dataset: str = "fsp_impedance_kapton",
    ) -> "RFChain":
        """Look up every component of the payload in the lab results."""
        coupler = payload_item(payload, "coupler")
        antenna = payload_item(payload, "antenna")
        try:
            attenuators = payload_items(payload, "attenuator")
        except ValueError:
            attenuators = []
        transmission = store.transmission(coupler, "transmission", frequency_ghz)
        coupling = store.transmission(coupler, "coupling_performance", frequency_ghz)
        gain = value_at(
            store.antenna_gain(antenna), frequency_ghz, "gain_dbi", "gain_err_db"
        )
        insertion_loss, _ = store.insertion_loss(antenna, frequency_ghz)
        reflection = store.reflection(antenna, frequency_ghz, dataset=impedance_dataset)
        return cls(
            frequency_ghz=frequency_ghz,
            coupler=coupler["name"],
            coupler_transmission_db=transmission[0],
            coupler_transmission_err_db=transmission[1],
            coupler_coupling_db=coupling[0],
            coupler_coupling_err_db=coupling[1],
            antenna=antenna["name"],
            antenna_gain_dbi=gain[0],
            antenna_gain_err_db=gain[1],
            antenna_insertion_loss_db=insertion_loss,
            antenna_reflection_db=reflection[0],
            antenna_reflection_err_db=reflection[1],
            attenuators=[
                {
                    "name": att["name"],
                    "attenuation_db": v,
                    "attenuation_err_db": e,
                }
                for att in attenuators
                for v, e in [store.transmission(att, "attenuation", frequency_ghz)]
            ],
        )

    @property
    def coupled_to_output_db(self) -> float:
        """Coupler output relative to the diode input: transmission - coupling."""
        return self.coupler_transmission_db - self.coupler_coupling_db

    @property
    def mismatch_loss_db(self) -> float:
        """Antenna mismatch loss 10 log10(1 - |S11|²)."""
        return float(linear_to_db(1 - db_to_linear(self.antenna_reflection_db)))

    @property
    def coupler_to_radiated_db(self) -> float:
        """Attenuators + antenna gain, insertion and mismatch losses."""
        attenuation = sum(a["attenuation_db"] for a in self.attenuators)
        return float(
            attenuation
            + self.antenna_gain_dbi
            + self.antenna_insertion_loss_db
            + self.mismatch_loss_db
        )

    def source_power(
        self, diode_input_mw: ArrayLike, diode_input_err_mw: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Radiated (EIRP) power on boresight from the diode input power.

        Returns dBm and its 1σ in mW; the uncertainties of the lab
        corrections are neglected (as in the original analysis).
        """
        coupler_out_dbm = mw_to_dbm(
            np.asarray(diode_input_mw) * db_to_linear(self.coupled_to_output_db)
        )
        coupler_out_err = np.asarray(diode_input_err_mw) * db_to_linear(
            self.coupled_to_output_db
        )
        power_dbm = coupler_out_dbm + self.coupler_to_radiated_db
        return power_dbm, coupler_out_err * db_to_linear(self.coupler_to_radiated_db)

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self) | {
            "mismatch_loss_db": self.mismatch_loss_db,
            "coupler_to_radiated_db": self.coupler_to_radiated_db,
        }


def free_space_loss_db(frequency_ghz: float, distance_m: ArrayLike) -> np.ndarray:
    """10 log10((λ / 4π d)²)."""
    wavelength = constants.c / (frequency_ghz * 1e9)
    return linear_to_db((wavelength / (4 * np.pi * np.asarray(distance_m, float))) ** 2)


def detector_power(
    source_dbm: ArrayLike,
    source_err_mw: ArrayLike,
    detector_gain_dbi: float,
    frequency_ghz: float,
    distance_m: ArrayLike,
    optical_efficiency: float = 1.0,
    duty_cycle: float = 2 / np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """Power on a telescope detector (dBm, 1σ in mW) from the source EIRP."""
    path_db = (
        detector_gain_dbi
        + free_space_loss_db(frequency_ghz, distance_m)
        + linear_to_db(optical_efficiency * duty_cycle)
    )
    return np.asarray(source_dbm) + path_db, np.asarray(source_err_mw) * db_to_linear(
        path_db
    )


def drone_emission_mw(
    distance_m: ArrayLike,
    solid_angle_sr: float,
    bandpass_ghz: float,
    temperature_k: float = 300.0,
    effective_diameter_m: float = 0.5,
    optical_efficiency: float = 1.0,
) -> np.ndarray:
    """Thermal power of the drone in the beam: k T B × beam dilution, in mW."""
    angular_diameter = effective_diameter_m / np.asarray(distance_m, float)
    dilution = np.pi * (angular_diameter / 2) ** 2 / solid_angle_sr
    k_mj = constants.k * 1e3  # mJ/K
    return k_mj * temperature_k * bandpass_ghz * 1e9 * optical_efficiency * dilution
