"""Tests for the column schema (pils.schema)."""

import polars as pl
import pytest

from pils import schema


def test_payload_gps_names_and_units():
    gps = pl.DataFrame(
        {
            "timestamp": [1.0],
            "posllh_lat": [-22.96],
            "posllh_lon": [-67.78],
            "posllh_height": [5179.7],
            "posllh_hMSL": [5148284],
            "posecef_ecefX": [222318137],
            "posllh_msgmode": [1],
        }
    )
    out = schema.canonical(gps, "gps")
    assert out.columns == [
        "timestamp",
        "lat_deg",
        "lon_deg",
        "h_ell_m",
        "h_msl_m",
        "ecef_x_m",
        "posllh_msg_mode",
    ]
    assert out["h_msl_m"][0] == pytest.approx(5148.284)  # mm -> m
    assert out["ecef_x_m"][0] == pytest.approx(2223181.37)  # cm -> m


def test_dji_drone_table_and_records():
    drone = pl.DataFrame(
        {"RTK:lat_p": [1.0], "RTK:hmsl_p": [5.0], "GPS:numSV": [12.0], "tick": [1]}
    )
    assert schema.canonical(drone, "dji").columns == [
        "rtk_lat_deg",
        "rtk_h_ell_m",
        "gps_num_sv",
        "tick",
    ]
    imu = pl.DataFrame(
        {"IMU_ATTI(0):gyroZ": [1.0], "IMU_ATTI(0):numSats": [3], "timestamp": [0.0]}
    )
    assert schema.canonical(imu, record="IMU_ATTI(0)").columns == [
        "gyro_z_deg_s",
        "num_sats",
        "timestamp",
    ]
    motor = pl.DataFrame({"Motor:Speed:RFront": [3800], "Motor:Status:RFront": [0]})
    assert schema.canonical(motor, record="Motor").columns == [
        "speed_rfront_rpm",
        "status_rfront",
    ]
    assert schema.record_name("IMU_ATTI(0)") == "imu_atti_0"
    assert schema.record_name("osd_data") == "osd"
    assert schema.record_name("BatteryInfo") == "battery_info"


def test_litchi_drops_raw_duplicates():
    litchi = pl.DataFrame(
        {
            "altitude(m)": [10.0],
            "yaw(deg)": [5.0],
            "yawRaw": [50.0],
            "gimbalYawAbs": [7.0],
        }
    )
    assert schema.canonical(litchi, "litchi").columns == [
        "h_rel_m",
        "yaw_deg",
        "gimbal_heading_deg",
    ]


def test_idempotent_and_generic():
    gps = pl.DataFrame({"posllh_hMSL": [5148284], "timeOfWeek_s": [1.0]})
    once = schema.canonical(gps, "gps")
    assert schema.canonical(once, "gps").equals(once)
    assert once.columns == ["h_msl_m", "time_of_week_s"]


def test_collision_is_an_error():
    df = pl.DataFrame({"RTK:lat_p": [1.0], "RTKdata:Lat_P": [1.0]})
    with pytest.raises(ValueError, match="rtk_lat_deg"):
        schema.canonical(df, "dji")


@pytest.mark.parametrize(
    "column, unit",
    [
        ("timestamp", "s"),
        ("rtk_lat_deg", "deg"),
        ("acc_x_m_s2", "m/s^2"),
        ("gyro_z_deg_s", "deg/s"),
        ("pressure_hpa", "hPa"),
        ("temperature_c", "degC"),
        ("amplitude_mv", "mV"),
        ("source_output_power_dbm", "dBm"),
        ("clock_drift_ns_s", "ns/s"),
        ("unix_time_ms", "ms"),
        ("ins_status", None),
        ("rtk_num_sv2", None),
        ("quat_w", None),
    ],
)
def test_unit_of(column, unit):
    assert schema.unit_of(column) == unit


def test_every_schema_name_is_snake_case():
    names = [f.name for table in schema.TABLES.values() for f in table.values()]
    names += [f.name for table in schema.DJI_RECORDS.values() for f in table.values()]
    assert all(schema.snake_case(n) == n for n in names)
