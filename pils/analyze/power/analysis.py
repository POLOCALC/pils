"""Expected source power on telescope detectors during a calibration flight.

Chain: payload diode reading (ADC) -> diode input power (lab responsivity at
the payload temperature) -> radiated source power (lab RF chain) -> pointing
loss of the source antenna towards the telescope -> free-space path and
detector beam -> power on a detector. The drone's own thermal emission in
the beam is computed alongside.
"""

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.interpolate import RegularGridInterpolator

from pils import store as result_store
from pils.analyze.power.rf_chain import (
    RFChain,
    detector_power,
    drone_emission_mw,
    payload_item,
)
from pils.analyze.power.signal import (
    DiodeCalibration,
    adc_envelope,
    bin_mean,
    scanning_range,
)
from pils.analyze.power.telescope import TelescopeBeam, load_config, telescope_beam
from pils.flight import Flight
from pils.lab_analysis.store import LabStore, value_at
from pils.store import new_version_name
from pils.utils import geometry
from pils.utils.logging_config import get_logger

logger = get_logger(__name__)

# ADC samples per half period of the source modulation, per band
ENVELOPE_WINDOW = {"WR10": 200, "WR06": 50}
SYNC_RATES = {
    "drone": 100.0,
    "litchi": 100.0,
    "inclinometer": 100.0,
    "payload": 100.0,
    "reference_gps": 100.0,
}
_BEAM_GRID_DEG = np.linspace(-90, 90, 501)


@dataclass
class PowerVersion:
    """Power analysis version container.

    Parameters
    ----------
    version_name : str
        ``rev_YYYYMMDD_HHMMSS``.
    power_data : pl.DataFrame
        One row per output sample (see :meth:`PowerAnalysis.run_analysis`).
    metadata : dict[str, Any]
        Inputs: beam, RF chain, scan range, settings.
    """

    version_name: str
    power_data: pl.DataFrame
    metadata: dict[str, Any]


class PowerAnalysis:
    """Expected power on a telescope's detectors from the payload source.

    File Structure
    --------------
    flight_dir/proc/power/          # see pils.store
    └── SATP1_ws0/                  # <TELESCOPE>_<wafer slot>
        └── rev_*/                  # manifest.json, power_data.parquet, plots

    Examples
    --------
    >>> flight = Flight(stout_flight)  # doctest: +SKIP
    >>> flight.add_sensor_data(["lm76", "adc", "inclinometer", "gps"])  # doctest: +SKIP
    >>> flight.add_drone_data()  # doctest: +SKIP
    >>> power = PowerAnalysis(flight)  # doctest: +SKIP
    >>> version = power.run_analysis("satp1", "ws0", save_data=True)  # doctest: +SKIP
    """

    def __init__(self, flight: Flight, output_dir: str | Path | None = None) -> None:
        """Analysis of a loaded flight; nothing is written until a save.

        Parameters
        ----------
        flight : Flight
            Flight with payload sensors (ADC, inclinometer, GPS, LM76 if
            any) and drone data loaded.
        output_dir : str | Path, optional
            Base folder for the results instead of the flight folder.
        """
        if not isinstance(flight, Flight):
            raise TypeError(f"Expected Flight object, got {type(flight).__name__}")
        if flight.flight_path is None:
            raise ValueError("Flight object must have a valid flight_path attribute")
        self.flight = flight
        self.flight_path = Path(flight.flight_path)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        base = self.output_dir or self.flight_path
        self.power_dir = base / "proc" / "power"

    @staticmethod
    def target_name(telescope: str, wafer_slot: str) -> str:
        """Folder of a telescope and wafer slot, e.g. ``SATP1_ws0``."""
        return f"{telescope.upper()}_{wafer_slot}"

    # -- versions -----------------------------------------------------------

    def list_versions(self, telescope: str, wafer_slot: str) -> list[str]:
        """List the runs for a telescope and wafer slot, oldest first."""
        return result_store.list_versions(
            self.power_dir, parent=self.target_name(telescope, wafer_slot)
        )

    def get_version(
        self, telescope: str, wafer_slot: str, version_name: str
    ) -> PowerVersion:
        """Load a saved run of a telescope and wafer slot."""
        tables, metadata = result_store.load_version(
            self.power_dir,
            version_name,
            parent=self.target_name(telescope, wafer_slot),
        )
        return PowerVersion(version_name, tables["power_data"], metadata)  # type: ignore[arg-type]

    def get_latest_version(
        self, telescope: str, wafer_slot: str
    ) -> PowerVersion | None:
        """Load the most recent run of a telescope and wafer slot, or None."""
        versions = self.list_versions(telescope, wafer_slot)
        return (
            self.get_version(telescope, wafer_slot, versions[-1]) if versions else None
        )

    # -- inputs -------------------------------------------------------------

    def _payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        if payload is not None:
            return payload
        payload_id = self.flight.flight_info.get("payload_id")
        if not payload_id:
            raise ValueError("No payload given and the flight has no payload_id")
        try:
            from stout.services.campaigns import CampaignService  # type: ignore
        except ImportError as e:
            raise ImportError("Pass the payload or install STOUT") from e
        found = CampaignService().get_payload_by_id(payload_id)
        if found is None:
            raise ValueError(f"Payload {payload_id} not found")
        return found

    def _frequency(self) -> float:
        info = self.flight.flight_info.get("flight_info") or {}
        if info.get("frequency") in (None, "", 0):
            raise ValueError(
                "The flight has no source frequency (flight_info.frequency)"
            )
        return float(info["frequency"])

    def _synced(self) -> dict[str, pl.DataFrame]:
        if self.flight.sync_data is None:
            self.flight.sync(
                use_rtk_data=True, common_time=True, target_rate=dict(SYNC_RATES)
            )
        sync = self.flight.sync_data
        drone = sync["drone"]
        valid = (
            drone["lat_deg"].is_not_null()
            & drone["lat_deg"].is_not_nan()
            & (drone["lat_deg"] != 0)
            & drone["lon_deg"].is_not_null()
        )
        return {
            k: (
                v.filter(valid)
                if isinstance(v, pl.DataFrame) and v.height == drone.height
                else v
            )
            for k, v in sync.items()
            if k != "reference_gps"
        }

    def _azel(
        self,
        telescope: str,
        azel_version: str | None,
        dji_broadcast_geod: dict[str, float],
        save: bool,
    ) -> tuple[pl.DataFrame, str]:
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(self.flight, output_dir=self.output_dir)
        if azel_version is not None:
            names = azel.list_versions(telescope_name=telescope)
            if azel_version not in names:
                raise ValueError(f"AZEL version {azel_version} not found ({names})")
            version = azel._load_version(azel_version, telescope_name=telescope)
        else:
            version = azel.run_analysis(telescope, dji_broadcast_geod, save_data=save)
        if version is None:
            raise ValueError(f"AZEL analysis failed for {telescope}")
        return version.azel_data, version.version_name

    @staticmethod
    def _temperature(
        sync: dict[str, pl.DataFrame], source: str | float
    ) -> tuple[np.ndarray | float, str]:
        if not isinstance(source, str):
            return float(source), "constant"
        if source in ("auto", "lm76") and "lm76" in sync:
            return sync["lm76"]["temperature_c"].to_numpy(), "lm76"
        if source in ("auto", "inclinometer") and "inclinometer" in sync:
            column = next(
                (c for c in ("temperature_c",) if c in sync["inclinometer"].columns),
                None,
            )
            if column is not None:
                return float(
                    np.nanmean(sync["inclinometer"][column].to_numpy())
                ), "inclinometer (scan mean)"
        raise ValueError(f"No temperature source '{source}' in the synchronized data")

    # -- run ----------------------------------------------------------------

    def run_analysis(
        self,
        telescope: str,
        wafer_slot: str,
        payload: dict[str, Any] | None = None,
        store: LabStore | None = None,
        bandpass: str = "gab",
        solid_angle_sr: float | None = None,
        temperature_source: str | float = "auto",
        azel_version: str | None = None,
        dji_broadcast_geod: dict[str, float] | None = None,
        output_rate_hz: float = 20.0,
        adc_rate_hz: float = 1600.0,
        adc_clip_sigma: float | None = None,
        impedance_dataset: str = "fsp_impedance_kapton",
        config: dict[str, Any] | None = None,
        save_data: bool = False,
        plot: bool = False,
    ) -> PowerVersion:
        """Compute the expected power on ``telescope``/``wafer_slot`` detectors.

        Parameters
        ----------
        telescope, wafer_slot : str
            Target (e.g. ``"satp1"``, ``"ws0"``); see the telescopes config.
        payload : dict, optional
            STOUT payload with items; looked up from the flight if None.
        store : LabStore, optional
            Lab results of the payload items.
        bandpass : str
            Detector bandpass setting (``"gab"``: 30 GHz, ``"SO"``: 26.9 GHz).
        solid_angle_sr : float, optional
            Detector beam solid angle, instead of the telescopes config.
        temperature_source : str | float
            ``"auto"`` (LM76, else inclinometer), ``"lm76"``,
            ``"inclinometer"`` or a constant in °C.
        azel_version : str, optional
            Use this saved AZEL version instead of computing one.
        dji_broadcast_geod : dict, optional
            DJI base position broadcast to the drone (config default).
        output_rate_hz, adc_rate_hz : float
            Output sampling and ADC sampling rates.
        adc_clip_sigma : float, optional
            Drop ADC samples above this many standard deviations.
        impedance_dataset : str
            Antenna impedance-matching set-up used for the mismatch loss.

        Returns
        -------
        PowerVersion
            ``power_data`` columns: ``timestamp``, ``expected_det_power_pw``,
            ``expected_det_power_err_pw``, ``drone_power_pw``,
            ``source_output_power_dbm`` (pointing corrected),
            ``source_output_power_err_mw``, ``source_boresight_power_dbm``,
            ``distance_m``, ``pointing_loss_db``, ``temperature_c``,
            ``diode_output_mv``, ``diode_output_err_mv``, ``diode_input_mw``,
            ``diode_input_err_mw``.
        """
        config = config or load_config()
        store = store or LabStore()
        payload = self._payload(payload)
        frequency = self._frequency()
        multiplier = payload_item(payload, "multiplier")
        band = multiplier["specifications"]["output_band"]
        if band not in ENVELOPE_WINDOW:
            raise ValueError(
                f"Band {band} not supported (have {sorted(ENVELOPE_WINDOW)})"
            )
        if bandpass not in config["bandpass_ghz"]:
            raise ValueError(
                f"Unknown bandpass '{bandpass}' ({sorted(config['bandpass_ghz'])})"
            )
        bandpass_ghz = float(config["bandpass_ghz"][bandpass])

        if solid_angle_sr is not None:
            beam = TelescopeBeam(
                telescope, wafer_slot, band, frequency, solid_angle_sr, "given"
            )
        else:
            beam = telescope_beam(telescope, wafer_slot, band, frequency, config)
        logger.info(f"Detector beam: {beam.as_metadata()}")

        # Synchronized data, scan range and geometry
        sync = self._synced()
        times = sync["drone"]["timestamp"].to_numpy()
        scan_start, scan_end = scanning_range(
            times, sync["drone"]["h_ell_m"].to_numpy(), config["scan_min_altitude_m"]
        )
        in_scan = (times > scan_start) & (times < scan_end)
        t_scan = times[in_scan]

        azel_data, azel_name = self._azel(
            telescope,
            azel_version,
            dji_broadcast_geod or config["dji_broadcast_geod"],
            save=save_data,
        )
        azel_t = azel_data["timestamp"].to_numpy()
        enu = np.array(
            [
                np.interp(t_scan, azel_t, azel_data[c].to_numpy())
                for c in ("e_m", "n_m", "u_m")
            ]
        )
        distance = np.linalg.norm(enu, axis=0)

        litchi = sync["litchi"].filter(pl.Series(in_scan))
        roll = sync["inclinometer"].filter(pl.Series(in_scan))["roll_deg"].to_numpy()
        yaw = (litchi["yaw_deg"] + litchi["gimbal_yaw_deg"]).to_numpy()
        pitch = litchi["gimbal_pitch_deg"].to_numpy()
        pointing_loss, rel_yaw, rel_pitch, poi = self._pointing_loss(
            enu, yaw, pitch, roll, store, payload_item(payload, "antenna"), frequency
        )

        temperature, temperature_label = self._temperature(
            {
                k: v.filter(pl.Series(in_scan))
                if isinstance(v, pl.DataFrame) and v.height == len(in_scan)
                else v
                for k, v in sync.items()
            },
            temperature_source,
        )

        # Lab calibration of the source
        calibration = DiodeCalibration.from_store(
            store, payload_item(payload, "detector"), frequency
        )
        rf = RFChain.from_payload(payload, frequency, store, impedance_dataset)

        # ADC: envelope of the modulated diode signal, binned to the output rate
        adc = getattr(self.flight.raw_data.payload_data, "adc", None)
        if adc is None:
            raise ValueError("No ADC data: flight.add_sensor_data('adc') first")
        adc = adc.filter(
            (pl.col("timestamp") > scan_start) & (pl.col("timestamp") < scan_end)
        )
        if adc_clip_sigma is not None:
            amp = adc["amplitude_mv"]
            adc = adc.filter(amp < adc_clip_sigma * amp.std())
        filtered, envelope, zero = adc_envelope(
            adc["amplitude_mv"].to_numpy(), ENVELOPE_WINDOW[band]
        )
        keep = ~np.isnan(envelope)
        adc_time = adc["timestamp"].to_numpy()[keep]
        factor = int(adc_rate_hz / output_rate_hz)
        t_bin, env_bin, env_err_bin = bin_mean(adc_time, envelope[keep], factor)

        t_out = np.arange(
            max(t_bin[0], t_scan[0]), min(t_bin[-1], t_scan[-1]), 1 / output_rate_hz
        )
        env = np.interp(t_out, t_bin, env_bin)
        env_err = np.interp(t_out, t_bin, env_err_bin)
        dist = np.interp(t_out, t_scan, distance)
        loss = np.interp(t_out, t_scan, pointing_loss)
        temp = (
            np.interp(t_out, t_scan, temperature)
            if isinstance(temperature, np.ndarray)
            else np.full(len(t_out), temperature)
        )

        diode_in, diode_in_err = calibration.input_power(env, env_err, temp)
        t_min, t_max = calibration.temperature_c[0], calibration.temperature_c[-1]
        clamped = float(np.mean((temp < t_min) | (temp > t_max)))
        if clamped:
            logger.warning(
                f"{clamped:.0%} of the samples are outside the diode calibration "
                f"({t_min:g} to {t_max:g} °C): the nearest calibration is used"
            )
        boresight_dbm, boresight_err_mw = rf.source_power(diode_in, diode_in_err)
        # TODO(confirm): the original analysis subtracts the pointing loss
        # (raising the power off-axis) while scaling the error down by it.
        source_dbm = boresight_dbm - loss
        source_err_mw = boresight_err_mw * 10 ** (loss / 10)

        det_dbm, det_err_mw = detector_power(
            source_dbm, source_err_mw, beam.gain_dbi, frequency, dist
        )
        drone_mw = drone_emission_mw(dist, beam.solid_angle_sr, bandpass_ghz)

        power_data = pl.DataFrame(
            {
                "timestamp": t_out,
                "expected_det_power_pw": 10 ** (det_dbm / 10) * 1e9,
                "expected_det_power_err_pw": det_err_mw * 1e9,
                "drone_power_pw": drone_mw * 1e9,
                "source_output_power_dbm": source_dbm,
                "source_output_power_err_mw": source_err_mw,
                "source_boresight_power_dbm": boresight_dbm,
                "distance_m": dist,
                "pointing_loss_db": loss,
                "temperature_c": temp,
                "diode_output_mv": env,
                "diode_output_err_mv": env_err,
                "diode_input_mw": diode_in,
                "diode_input_err_mw": diode_in_err,
            }
        )
        version = PowerVersion(
            version_name=new_version_name(),
            power_data=power_data,
            metadata={
                "flight": self.flight.flight_info.get("flight_name"),
                "frequency_ghz": frequency,
                "band": band,
                "payload": payload.get("name"),
                "beam": beam.as_metadata(),
                "bandpass": bandpass,
                "bandpass_ghz": bandpass_ghz,
                "rf_chain": rf.as_metadata(),
                "diode_calibration": {
                    "temperatures_c": calibration.temperature_c,
                    "control_voltages_v": calibration.control_voltage_v,
                },
                "temperature_source": temperature_label,
                "temperature_outside_calibration_fraction": clamped,
                "azel_version": azel_name,
                "scan_start": scan_start,
                "scan_end": scan_end,
                "output_rate_hz": output_rate_hz,
                "adc_rate_hz": adc_rate_hz,
                "adc_clip_sigma": adc_clip_sigma,
                "envelope_window": ENVELOPE_WINDOW[band],
                "impedance_dataset": impedance_dataset,
                "created": datetime.datetime.now().isoformat(),
            },
        )
        if save_data:
            path = result_store.save_version(
                self.power_dir,
                version.version_name,
                {"power_data": power_data},
                version.metadata,
                parent=self.target_name(telescope, wafer_slot),
                kind="power",
            )
            logger.info(f"Saved power version {version.version_name} to {path}")
        if plot:
            self._save_plots(
                version,
                telescope,
                wafer_slot,
                {
                    "t_scan": t_scan,
                    "distance": distance,
                    "poi": poi,
                    "rel_yaw": rel_yaw,
                    "rel_pitch": rel_pitch,
                    "pointing_loss": pointing_loss,
                    "adc_time": adc["timestamp"].to_numpy(),
                    "adc_amplitude": adc["amplitude_mv"].to_numpy(),
                    "adc_filtered": filtered,
                    "adc_envelope": envelope,
                    "adc_zero": zero,
                },
            )
        return version

    @staticmethod
    def _pointing_loss(
        enu: np.ndarray,
        yaw: np.ndarray,
        pitch: np.ndarray,
        roll: np.ndarray,
        store: LabStore,
        antenna: dict[str, Any],
        frequency: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Source antenna gain loss (dB) towards the telescope at the origin."""
        poi = geometry.compute_poi(enu[0], enu[1], enu[2], yaw, pitch, roll)
        rel_yaw = np.zeros(len(poi))
        rel_pitch = np.zeros(len(poi))
        for i in range(len(poi)):
            frame = geometry.build_frame(enu[:, i], poi[i], roll[i])
            telescope_local = geometry.world_to_local(frame, np.zeros(3))
            poi_local = geometry.world_to_local(frame, poi[i])
            rel_yaw[i], rel_pitch[i], _ = geometry.relative_yaw_pitch_roll(
                telescope_local, poi_local
            )

        fwhm = store.antenna_fwhm(antenna)

        def plane_fwhm(plane: str) -> float:
            part = fwhm.filter(pl.col("plane") == plane)
            return value_at(part, frequency, "fwhm_deg", "fwhm_err_deg")[0]

        grid = _BEAM_GRID_DEG
        beam = geometry.beam_2d_gauss(
            grid, grid, plane_fwhm("Hplane"), plane_fwhm("Eplane")
        )
        interp = RegularGridInterpolator(
            (grid, grid), beam, method="linear", bounds_error=False, fill_value=0.0
        )
        # TODO(confirm): as in the original analysis the grid is indexed
        # (E, H) but queried with (yaw, pitch), i.e. yaw offsets use the
        # E-plane width and pitch offsets the H-plane width.
        points = np.column_stack([-np.degrees(rel_yaw), -np.degrees(rel_pitch)])
        with np.errstate(divide="ignore"):
            loss = 10 * np.log10(interp(points))
        return loss, rel_yaw, rel_pitch, poi

    def _save_plots(
        self,
        version: PowerVersion,
        telescope: str,
        wafer_slot: str,
        extra: dict[str, np.ndarray],
    ) -> list[Path]:
        from matplotlib.figure import Figure

        folder = (
            self.power_dir
            / self.target_name(telescope, wafer_slot)
            / version.version_name
        )
        folder.mkdir(parents=True, exist_ok=True)
        data = version.power_data
        t0 = data["timestamp"][0]
        paths = []

        def save(fig: Any, name: str) -> None:
            paths.append(folder / f"{name}.png")
            fig.savefig(paths[-1], dpi=150)

        fig = Figure(figsize=(14, 4), layout="constrained")
        ax_poi, ax_off, ax_loss = fig.subplots(1, 3)
        ax_poi.scatter(extra["poi"][:, 0], extra["poi"][:, 1], s=0.1)
        ax_poi.scatter(0, 0, color="red", label=telescope)
        ax_poi.set(
            xlabel="E [m]", ylabel="N [m]", title="Source pointing on the ground"
        )
        ax_poi.legend()
        ax_off.scatter(
            -np.degrees(extra["rel_yaw"]), -np.degrees(extra["rel_pitch"]), s=2
        )
        ax_off.set(
            xlabel="Yaw offset [deg]",
            ylabel="Pitch offset [deg]",
            title="Telescope in the source beam",
        )
        ax_loss.plot(extra["t_scan"] - t0, extra["pointing_loss"], color="black")
        ax_loss.set(xlabel="Time [s]", ylabel="[dB]", title="Pointing gain loss")
        save(fig, "pointing")

        fig = Figure(figsize=(10, 4), layout="constrained")
        ax = fig.subplots()
        t = extra["adc_time"] - t0
        ax.plot(t, extra["adc_amplitude"], ls="--", lw=0.5, label="raw")
        ax.plot(t, extra["adc_filtered"], lw=0.5, label="filtered")
        ax.plot(t, extra["adc_envelope"], label="envelope")
        ax.plot(t, extra["adc_zero"], color="red", label="zero level")
        ax.set(xlabel="Time [s]", ylabel="Voltage [mV]", title="Diode output")
        ax.legend()
        save(fig, "adc")

        fig = Figure(figsize=(12, 8), layout="constrained")
        axes = fig.subplots(2, 2).flat
        t = data["timestamp"] - t0
        panels = [
            ("distance_m", None, "Distance [m]", f"Distance drone-{telescope}"),
            ("temperature_c", None, "Temperature [°C]", "Payload temperature"),
            ("source_output_power_dbm", None, "Power [dBm]", "Source output power"),
            (
                "expected_det_power_pw",
                "expected_det_power_err_pw",
                "Power [pW]",
                "Expected power on detectors",
            ),
        ]
        for ax, (column, err, label, title) in zip(axes, panels, strict=True):
            ax.plot(t, data[column], color="black")
            if err:
                ax.fill_between(
                    t,
                    data[column] - data[err],
                    data[column] + data[err],
                    alpha=0.4,
                    color="gold",
                )
            ax.set(xlabel="Time [s]", ylabel=label, title=title)
        save(fig, "power")
        logger.info(f"Saved power plots to {folder}")
        return paths
