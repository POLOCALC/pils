"""DJI Matrice 400: the ``psdk_telemetry`` stream format (``uav_telemetry.bin``).

Ported (copied and adapted, not imported) from LAGER, POLOCALC/lager at 825164a:
``lager/UAVs/DJI_M400/parameters.py``, only the constants the decoding needs
(the stream format of ``application/telemetry_stream.h``).
"""

TELEMETRY_FILENAME = "uav_telemetry.bin"

SYNC = b"\xa5\x5a"
HEADER_FMT = "<BBHHqII"  # after SYNC: version, type, topic, length, host ns, aircraft ms, aircraft us
HEADER_SIZE = 24  # SYNC + HEADER_FMT
CRC_SIZE = 2
MAX_PAYLOAD = 2048  # TELEMETRY_STREAM_MAX_PAYLOAD
STREAM_VERSION = 1
TOPIC_NAME_SIZE = 32
TOPIC_ENTRY_SIZE = (
    7 + TOPIC_NAME_SIZE
)  # u16 topic, u16 Hz, u16 size, u8 subscribed, char name[32]

RECORD_DATA, RECORD_STATUS, RECORD_TOPIC_TABLE, RECORD_STATS = 1, 2, 3, 4
STATE_NAMES = {
    1: "starting",
    2: "connecting",
    3: "connected",
    4: "subscribed",
    5: "stopping",
    10: "info",
    11: "warning",
    100: "error",
}

DOP_SCALE = 0.01  # hdop/pdop unit (PSDK header: 100 = 1.00)
