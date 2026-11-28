"""Synchronizing the telemetry, gimbal log and flight record of a drone with the payload GPS."""

import numpy as np
import polars as pl
import pytest

from pils.drones.sources import telemetry_positions
from pils.flight import DroneData, Flight, PayloadData
from pils.synchronizer import Synchronizer

DURATION = 200.0


def track(t):
    """A manoeuvring track (turns of 10 s to 90 s): latitude, longitude, ellipsoidal height at times t (s)."""
    lat = (
        45.0 + 0.0004 * np.sin(0.07 * t) + 0.00006 * np.sin(0.9 * t) + 0.00002 * t / 10
    )
    lon = 9.0 + 0.0005 * np.cos(0.05 * t) + 0.00005 * np.sin(0.7 * t + 1)
    h = 200.0 + 15.0 * np.sin(0.04 * t) + 5.0 * np.sin(0.45 * t)
    return lat, lon, h


def payload_gps():
    t = np.arange(0, DURATION, 0.2)
    lat, lon, h = track(t)
    return pl.DataFrame(
        {
            "timestamp": t,
            "lat_deg": lat,
            "lon_deg": lon,
            "h_ell_m": h,
            "h_msl_m": h - 30.0,
        }
    )


def telemetry(offset: float, rtk_valid=True, fill=True):
    """LAGER-like telemetry on a clock ``offset`` s ahead: RTK at 5 Hz among rows of other topics at 10 Hz."""
    t = np.arange(0, DURATION, 0.1)
    row_t = t + offset
    lat, lon, h = track(t)
    rtk_epoch = (
        np.arange(len(t)) % 2 == 0
    )  # the rtk_position topic arrives every other row
    frame = pl.DataFrame(
        {
            "timestamp": row_t,
            "topic": np.where(rtk_epoch, "rtk_position", "quaternion"),
            "rtk_lat_deg": np.where(rtk_epoch, lat, np.nan) if rtk_valid else 0.0,
            "rtk_lon_deg": np.where(rtk_epoch, lon, np.nan) if rtk_valid else 0.0,
            "rtk_h_ell_m": np.where(rtk_epoch, h, np.nan) if rtk_valid else 0.0,
            "yaw_deg": 90.0 + t * 0.1,
        }
    ).with_columns(pl.col("rtk_lat_deg", "rtk_lon_deg", "rtk_h_ell_m").fill_nan(None))
    return (
        frame.with_columns(
            pl.col("rtk_lat_deg", "rtk_lon_deg", "rtk_h_ell_m").forward_fill()
        )
        if fill
        else frame
    )


def gimbal(offset: float):
    t = np.arange(0, DURATION, 0.5) + offset
    return {
        "mount_orientation": pl.DataFrame(
            {"timestamp": t, "pitch_deg": -30.0 + 0.0 * t}
        )
    }


def flight_with(drone: DroneData):
    flight = Flight(
        {"drone_data_folder_path": "/x/drone", "aux_data_folder_path": "/x/aux"}
    )
    flight._drone_model = "dji"
    flight.raw_data.payload_data = PayloadData()
    flight.raw_data.payload_data.gps = payload_gps()
    flight.raw_data.drone_data = drone
    return flight


def test_a_matrice_400_without_dat_takes_its_telemetry_as_the_drone_gps():
    flight = flight_with(
        DroneData(sources={"uav_telemetry": telemetry(2.0), "gimbal": gimbal(2.0)})
    )
    synced = flight.sync(target_rate={"drone": 5.0})
    assert flight.sync_offsets["drone_gps"] == pytest.approx(-2.0, abs=0.15)
    assert {"drone", "reference_gps", "gimbal_mount_orientation"} <= set(synced)
    assert {"lat_deg", "lon_deg", "h_ell_m"} <= set(synced["drone"].columns)
    # the gimbal is on the telemetry's clock: the same offset, on the common grid
    assert synced["gimbal_mount_orientation"]["timestamp"][0] == pytest.approx(
        synced["drone"]["timestamp"][0]
    )
    assert flight.sync_position_columns["drone"]["lat"] == "rtk_lat_deg"


def test_telemetry_next_to_a_dat_is_a_source_of_its_own():
    dat = pl.DataFrame({"timestamp": np.arange(0, DURATION, 0.1) + 1.0}).with_columns(
        rtk_lat_deg=pl.Series(track(np.arange(0, DURATION, 0.1))[0]),
        rtk_lon_deg=pl.Series(track(np.arange(0, DURATION, 0.1))[1]),
        rtk_h_ell_m=pl.Series(track(np.arange(0, DURATION, 0.1))[2]),
    )
    flight = flight_with(
        DroneData(
            dat, sources={"uav_telemetry": telemetry(-3.0), "gimbal": gimbal(-3.0)}
        )
    )
    synced = flight.sync(target_rate={"drone": 5.0})
    assert flight.sync_offsets["drone_gps"] == pytest.approx(
        -1.0, abs=0.15
    )  # the .DAT is the drone GPS
    assert flight.sync_offsets["telemetry_gps"] == pytest.approx(
        3.0, abs=0.15
    )  # telemetry: its own clock
    assert {"drone", "telemetry", "gimbal_mount_orientation"} <= set(synced)
    assert "rtk_lat_deg" in synced["telemetry"].columns  # its own column names
    assert flight.sync_offsets["telemetry_gps"] != flight.sync_offsets["drone_gps"]


def test_the_flight_record_is_synchronized_on_its_own_clock():
    t = np.arange(0, DURATION, 0.1)
    lat, lon, h = track(t)
    record = pl.DataFrame(
        {
            "timestamp": t + 0.5,
            "lat_deg": lat,
            "lon_deg": lon,
            "h_m": h + 40.0,
            "h_rel_m": h - 200,
        }
    )
    flight = flight_with(
        DroneData(sources={"uav_telemetry": telemetry(2.0), "flight_record": record})
    )
    synced = flight.sync(target_rate={"drone": 5.0})
    assert flight.sync_offsets["flight_record_gps"] == pytest.approx(-0.5, abs=0.15)
    assert "flight_record" in synced and "lat_deg" in synced["flight_record"].columns
    # the unknown reference of its height is not turned into an ellipsoidal one
    assert "h_ell_m" not in synced["flight_record"].columns


def test_telemetry_without_a_receiver_fix_is_not_the_drone_gps():
    flight = flight_with(
        DroneData(sources={"uav_telemetry": telemetry(2.0, rtk_valid=False)})
    )
    synced = flight.sync(target_rate={"drone": 5.0})
    assert "drone" not in synced and "reference_gps" in synced  # the payload GPS alone


def test_the_positions_are_the_epochs_of_the_receiver_not_the_carried_values():
    table = telemetry(0.0)
    assert table.height == 2000
    frame, cols = telemetry_positions(table, receivers_only=True)
    assert frame.height == 1000 and cols["alt"] == "rtk_h_ell_m"
    assert frame["timestamp"].diff().drop_nulls().to_numpy() == pytest.approx(0.2)
    assert (
        telemetry_positions(telemetry(0.0, rtk_valid=False), receivers_only=True)
        is None
    )
    # the fused position and the app's are not receivers
    fused = pl.DataFrame(
        {
            "timestamp": [1.0, 2.0],
            "fused_lat_deg": [45.0, 45.1],
            "fused_lon_deg": [9.0, 9.1],
            "fused_h_m": [1.0, 2.0],
        }
    )
    assert telemetry_positions(fused, receivers_only=True) is None
    assert telemetry_positions(fused)[1]["lat"] == "fused_lat_deg"


def test_a_position_source_and_attached_table_in_the_synchronizer():
    sync = Synchronizer()
    ref = payload_gps()
    sync.add_gps_reference(ref, lat_col="lat_deg", lon_col="lon_deg", alt_col="h_ell_m")
    src = telemetry(1.5).filter(pl.col("topic") == "rtk_position")
    sync.add_position_source(
        "telemetry", src, "rtk_lat_deg", "rtk_lon_deg", "rtk_h_ell_m"
    )
    sync.add_attached_table(
        "gimbal_x", gimbal(1.5)["mount_orientation"], to="telemetry_gps"
    )
    sync.add_attached_table(
        "orphan", gimbal(0.0)["mount_orientation"], to="nothing_gps"
    )
    out = sync.synchronize(target_rate={"drone": 5.0})
    assert sync.offsets["telemetry_gps"]["time_offset"] == pytest.approx(-1.5, abs=0.15)
    assert {"telemetry", "gimbal_x"} <= set(out) and "orphan" not in out
    assert sync.rates["telemetry"] == 5.0
    with pytest.raises(ValueError, match="missing columns"):
        sync.add_position_source("bad", src, "nope", "rtk_lon_deg", "rtk_h_ell_m")
    with pytest.raises(ValueError, match="empty"):
        sync.add_position_source(
            "bad", src.head(0), "rtk_lat_deg", "rtk_lon_deg", "rtk_h_ell_m"
        )


def test_the_synchronized_sources_are_saved_and_loaded_with_their_offsets(tmp_path):
    flight = flight_with(
        DroneData(sources={"uav_telemetry": telemetry(2.0), "gimbal": gimbal(2.0)})
    )
    flight.sync(target_rate={"drone": 5.0})
    revision = flight.save(tmp_path / "flight")
    sync_dir = tmp_path / "flight" / "sync" / revision
    assert (sync_dir / "gimbal_mount_orientation.parquet").is_file() and (
        sync_dir / "drone.parquet"
    ).is_file()
    back = Flight.load(tmp_path / "flight")
    assert back.sync_offsets["drone_gps"] == pytest.approx(
        flight.sync_offsets["drone_gps"]
    )
    assert back.sync_data["gimbal_mount_orientation"].equals(
        flight.sync_data["gimbal_mount_orientation"]
    )


def flight_record(offset: float = 0.0, rtk=True):
    """A Matrice 400 flight record (schema names): 10 Hz frames, RTK once a second."""
    t = np.arange(0, DURATION, 0.1)
    lat, lon, h = track(t)
    rtk_epoch = np.arange(len(t)) % 10 == 0
    frame = pl.DataFrame(
        {
            "timestamp": t + offset,
            "lat_deg": lat,
            "lon_deg": lon,
            "h_m": h - 300.0,  # barometric: unknown reference
            "abs_lat_deg": lat,
            "abs_lon_deg": lon,
            "abs_h_m": h,
            "yaw_deg": 90.0 + t * 0.1,
        }
    )
    if rtk:
        frame = frame.with_columns(
            rtk_lat_deg=pl.Series(np.where(rtk_epoch, lat, np.nan)),
            rtk_lon_deg=pl.Series(np.where(rtk_epoch, lon, np.nan)),
            rtk_h_ell_m=pl.Series(np.where(rtk_epoch, h, np.nan)),
            rtk_pos_flag_1=pl.Series(np.where(rtk_epoch, 50, 0)),
            rtk_base_lat_deg=pl.Series(np.where(rtk_epoch, 45.0001, np.nan)),
            rtk_base_lon_deg=pl.Series(np.where(rtk_epoch, 9.0001, np.nan)),
            rtk_base_h_ell_m=pl.Series(np.where(rtk_epoch, 190.0, np.nan)),
        ).with_columns(pl.col("^rtk_.*_(deg|m)$").fill_nan(None))
    return frame


def test_a_flight_record_alone_is_the_drone_gps_through_its_rtk():
    flight = flight_with(DroneData(sources={"flight_record": flight_record()}))
    flight.raw_data.payload_data.gps = None  # no payload GPS either
    synced = flight.sync(target_rate={"drone": 5.0})
    assert flight.sync_reference == "drone_gps"
    assert flight.sync_position_columns["drone"]["lat"] == "rtk_lat_deg"
    # the whole record (10 Hz) on the drone GPS's clock, not as a source of its own
    assert "flight_record" in synced and "flight_record_gps" not in flight.sync_offsets
    assert synced["flight_record"]["timestamp"][0] == pytest.approx(
        synced["drone"]["timestamp"][0]
    )


def test_a_flight_record_without_rtk_is_not_the_drone_gps():
    flight = flight_with(DroneData(sources={"flight_record": flight_record(rtk=False)}))
    flight.raw_data.payload_data.gps = None
    with pytest.raises(ValueError, match="No reference timebase"):
        flight.sync(target_rate={"drone": 5.0})


def test_a_flight_record_next_to_the_payload_gps_keeps_its_rate():
    flight = flight_with(DroneData(sources={"flight_record": flight_record(0.5)}))
    synced = flight.sync(target_rate={"drone": 5.0})
    # the RTK makes the record the drone GPS; offset found against the payload GPS
    assert flight.sync_offsets["drone_gps"] == pytest.approx(-0.5, abs=0.15)
    assert "flight_record" in synced


def test_the_flight_record_as_a_source_uses_the_positions_of_every_frame():
    from pils.drones.sources import record_positions

    frame, cols = record_positions(flight_record())
    assert cols == {"lat": "abs_lat_deg", "lon": "abs_lon_deg", "alt": "abs_h_m"}
    assert frame.height == int(DURATION / 0.1)


def test_the_broadcast_base_comes_from_the_flight_record():
    from pils.drones.sources import record_rtk_base

    base = record_rtk_base(flight_record())
    assert base == pytest.approx({"lat": 45.0001, "lon": 9.0001, "alt": 190.0})
    assert record_rtk_base(flight_record(rtk=False)) is None
    assert record_rtk_base(None) is None
