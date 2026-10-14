"""Command line for the lab and telescope analyses.

Examples
--------
::

    pils lab vna /mnt/data/POLOCALC/lab/WR10/attenuator/ATT_002/attenuation
    pils lab beam .../antenna/ANT_001/radiation_pattern --nominal-gain 15
    pils lab mult-diode .../detector/DET_001/responsivity
    pils power flight_20251201_1515 -t satp1 -w ws0

Runs are saved (results file + plots) unless ``--dry-run``; ``--output-dir``
writes them somewhere else than the data folders.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from pils.utils.logging_config import get_logger

logger = get_logger(__name__)


def _print_table(title: str, table: Any) -> None:
    print(f"\n{title}")
    print(table)


def _lab(args: argparse.Namespace) -> int:
    from pils.lab_analysis import (
        BeamAnalysis,
        LabStore,
        MultiplierDiodeAnalysis,
        VNAAnalysis,
    )

    save = not args.dry_run
    plot = save and not args.no_plot
    common = {"stem": args.stem, "output_dir": args.output_dir}
    if args.kind == "vna":
        vna = VNAAnalysis.from_folder(args.folder, **common)
        version = vna.run_analysis(corrected=args.corrected, save_data=save, plot=plot)
        _print_table("S parameters per frequency", version["summary"])
    elif args.kind == "beam":
        beam = BeamAnalysis.from_folder(args.folder, **common)
        version = beam.run_analysis(
            nominal_gain_dbi=args.nominal_gain,
            n_fwhm=args.n_fwhm,
            store=LabStore(args.lab_root),
            save_data=save,
            plot=plot,
        )
        _print_table("Gain", version["gain"])
    else:
        run = MultiplierDiodeAnalysis.from_folder(args.folder, **common)
        version = run.run_analysis(
            store=LabStore(args.lab_root), save_data=save, plot=plot
        )
        _print_table("Corrections", version["corrections"])
    state = "saved" if save else "not saved (dry run)"
    print(f"\n{version.version_name} {state}")
    return 0


def _power(args: argparse.Namespace) -> int:
    from pils.analyze.power import PowerAnalysis
    from pils.flight import Flight
    from pils.lab_analysis import LabStore
    from pils.loader.stout import StoutLoader

    meta = StoutLoader().load_single_flight(flight_name=args.flight)
    if meta is None:
        print(f"Flight {args.flight} not found", file=sys.stderr)
        return 1
    flight = Flight(meta)
    sensors_dir = Path(meta["aux_data_folder_path"]) / "sensors"
    sensors = ["adc", "inclinometer", "gps"]
    if any(sensors_dir.glob("*TMP*.csv")):  # LM76 (SENSOR_MAP pattern)
        sensors.insert(0, "lm76")
    flight.add_sensor_data(sensors)
    flight.add_drone_data()

    temperature: str | float = args.temperature
    try:
        temperature = float(args.temperature)
    except ValueError:
        pass
    analysis = PowerAnalysis(flight, output_dir=args.output_dir)
    version = analysis.run_analysis(
        args.telescope,
        args.wafer_slot,
        store=LabStore(args.lab_root),
        bandpass=args.bandpass,
        solid_angle_sr=args.solid_angle,
        temperature_source=temperature,
        azel_version=args.azel_version,
        save_data=not args.dry_run,
        plot=not args.dry_run and not args.no_plot,
    )
    _print_table(
        "Expected power on detectors (pW)",
        version.power_data["expected_det_power_pw"].describe(),
    )
    state = "saved" if not args.dry_run else "not saved (dry run)"
    print(f"\n{version.version_name} {state}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pils", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def outputs(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--output-dir", help="Save results here instead of the data folder"
        )
        p.add_argument("--dry-run", action="store_true", help="Compute without saving")
        p.add_argument("--no-plot", action="store_true", help="Do not save plots")
        p.add_argument(
            "--lab-root", help="Lab data root (default: PILS_LAB_DATA_PATH or STOUT)"
        )

    lab = sub.add_parser("lab", help="Process a lab measurement")
    lab_sub = lab.add_subparsers(dest="kind", required=True)
    for kind, help_text in [
        ("vna", "VNA two-port S parameters (.s2p)"),
        ("beam", "Antenna beam pattern: FWHM and gain"),
        ("mult-diode", "Multiplier output power and diode responsivity"),
    ]:
        p = lab_sub.add_parser(kind, help=help_text)
        p.add_argument(
            "folder", help="Test folder: <lab>/<band>/<category>/<item>/<test_type>"
        )
        p.add_argument(
            "--stem", help="Measurement (YYYYMMDD_HHMMSS); latest if omitted"
        )
        outputs(p)
        if kind == "vna":
            p.add_argument(
                "--corrected",
                action="store_true",
                help="Mismatch-corrected transmission",
            )
        if kind == "beam":
            p.add_argument("--nominal-gain", type=float, help="dBi; default from STOUT")
            p.add_argument(
                "--n-fwhm", type=float, default=1.0, help="Fit half-width in FWHMs"
            )

    power = sub.add_parser("power", help="Expected power on telescope detectors")
    power.add_argument("flight", help="Flight name, e.g. flight_20251201_1515")
    power.add_argument("-t", "--telescope", required=True)
    power.add_argument("-w", "--wafer-slot", required=True)
    power.add_argument(
        "-s", "--bandpass", default="gab", help="Detector setting: gab or SO"
    )
    power.add_argument(
        "--solid-angle", type=float, help="Detector beam solid angle (sr)"
    )
    power.add_argument(
        "--temperature", default="auto", help="auto, lm76, inclinometer or °C"
    )
    power.add_argument("--azel-version", help="Use a saved AZEL version")
    outputs(power)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "lab":
        return _lab(args)
    return _power(args)


if __name__ == "__main__":
    sys.exit(main())
