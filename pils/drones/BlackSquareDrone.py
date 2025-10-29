import numpy as np
import pandas as pd
from collections import defaultdict

from astropy.utils.iers import LeapSeconds

"""https://ardupilot.org/copter/docs/logmessages.html"""

ARDUTYPES = {
    'a': (np.int16, (32,)),        # int16_t[32]
    'b': np.int8,                  # int8_t
    'B': np.uint8,                 # uint8_t
    'h': np.int16,                 # int16_t
    'H': np.uint16,                # uint16_t
    'i': np.int32,                 # int32_t
    'I': np.uint32,                # uint32_t
    'f': np.float32,               # float
    'd': np.float64,               # double
    'n': 'S4',                     # char[4]
    'N': 'S16',                    # char[16]
    'Z': 'S64',                    # char[64]
    'c': np.float64, #np.int16,                 # int16_t * 100, usually scaled
    'C': np.float64, #np.uint16,                # uint16_t * 100, usually scaled
    'e': np.float64, #np.int32,                 # int32_t * 100, usually scaled
    'E': np.float64, #np.uint32,                # uint32_t * 100, usually scaled
    'L': np.float64, #np.int32,                 # int32_t * 1e7 latitude/longitude
    'M': "S64", #np.uint8,                 # uint8_t flight mode
    'q': np.int64,                 # int64_t
    'Q': np.uint64,                # uint64_t
}
ARDUFACTOR = {
    'c' : 100, 
    'C' : 100, 
    'e' : 100, 
    'E' : 100, 
    'L' : 1e7,
}
FLIGHTMODES = {
    "Stabilize": 0,
    "Acro": 1,
    "Altitude Hold": 2,
    "Auto": 3,
    "Guided": 4,
    "Loiter": 5,
    "RTL": 6,
    "Circle": 7,
    "Land": 9,
    "Drift": 11,
    "Sport": 13,
    "Flip": 14,
    "AutoTune": 15,
    "PosHold": 16,
    "Brake": 17,
    "Throw": 18,
    "Avoid_ADSB": 19,
    "Guided_NoGPS": 20,
    "Smart RTL": 21,
    "FlowHold": 22,
    "Follow": 23,
    "ZigZag": 24,
    "System Identification": 25,
    "Heli_Autorotate": 26,
    "Turtle": 27
}


def messages_to_df(messages, columns, format_str):
    dtypes = []
    for col, f in zip(columns, format_str):
        np_dtype = ARDUTYPES.get(f, object)
        dtypes.append((col, np_dtype))

    # Create structured array
    arr = np.array([tuple(row) for row in messages], dtype=dtypes)
    return pd.DataFrame(arr)

def read_msgs(path):
    with open(path, "r") as f:
        lines = (line.strip() for line in f)

        # First pass: extract formats and group messages
        formats = {}
        grouped_msgs = defaultdict(list)

        for line in lines:
            #Extract all the formats
            if line.startswith("FMT"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    _, _, _, msg_type, format_str, *colnames = parts
                    formats[msg_type] = {"Format": format_str, "Columns": colnames}
            #Extract all the messages
            elif line and not line.startswith("FILE"):
                parts = [p.strip() for p in line.split(",")]
                msg_type, values = parts[0], parts[1:]
                grouped_msgs[msg_type].append(values)

    #Start parsing
    dfs = {}
    for msg_type, messages in grouped_msgs.items():
        if msg_type not in formats:
            continue

        columns = formats[msg_type]["Columns"]
        format_str = formats[msg_type]["Format"]

        try:
            df = messages_to_df(messages, columns, format_str)
            dfs[msg_type] = df.reset_index(drop=True)
        except Exception as e:
            print(f"Warning: Failed to parse message type '{msg_type}': {e}")

    return dfs

def generate_log_file(lines):
    msgs = []
    for line in lines:
        msg = line.strip().split(",")
        if msg[0] == "FILE":
            msgs.append([msg[1], msg[4][1:]])

    msg_df = pd.DataFrame(msgs, columns=["Name", "Log"])
    names = msg_df["Name"].unique()
    s = msg_df.groupby("Name").get_group(names[0])["Log"].sum()
    decoded_str = s.encode('utf-8').decode('unicode_escape')

    # Write to a text file with proper formatting
    with open("test.txt", "w", encoding="utf-8") as f:
        f.write(decoded_str)

def get_leapseconds(year, month):
# Load and prepare DataFrame
    ls_df = LeapSeconds.auto_open().to_pandas()
    ls_df["date"] = pd.to_datetime(ls_df[['year', 'month', 'day']])
    ls_df = ls_df.set_index("date").sort_index()

    # Define dates
    start_date = pd.Timestamp("1980-01-01")
    end_date = pd.Timestamp(f"{year}-{month:02d}-01")

    # Use asof to find nearest previous dates
    tai_utc_start = ls_df.loc[ls_df.index.asof(start_date), "tai_utc"]
    tai_utc_end = ls_df.loc[ls_df.index.asof(end_date), "tai_utc"]

    # Compute leap seconds
    leap_seconds = tai_utc_end - tai_utc_start
    return leap_seconds

class BlackSquareDrone:
    def __init__(self, path):

        self.path = path

        self.data = None
        self.imu = None
        self.barometer = None
        self.magnetometer = None
        self.gps = None
        self.batteries = None
        self.attitude = None
        self.pwm = None
        self.position = None
        self.gpa = None
        self.gimbal = None
        self.params = None

        self.datetime = None

    def load_data(self):
        self.data = read_msgs(self.path)
        self.imu = self.data["IMU"]
        self.barometer = self.data["BARO"]
        self.magnetometer = self.data["MAG"]
        self.gps = self.data["GPS"]
        self.batteries = self.data["BAT"]
        self.attitude = self.data["ATT"]
        self.pwm = self.data["RCOU"]
        self.position = self.data["POS"]
        self.gpa = self.data["GPA"]

        self.params = self.data["PARM"]
        self.params["Name"] = self.params["Name"].apply(lambda x: x.decode("utf-8"))
        self.params = self.params.set_index("Name")

        if "MNT" in self.data.keys():
            self.gimbal = self.data["MNT"]

    def compute_datetime(self):
        gps = self.gps#.groupby("I").get_group(2)
        gps_dt = pd.to_datetime("1980-01-01 00:00:00") + pd.to_timedelta(gps["GWk"], unit='w') + pd.to_timedelta(gps["GMS"], unit='ms')
        leapseconds = get_leapseconds(gps_dt.dt.year[0], gps_dt.dt.month[0])
        gps_dt -= pd.Timedelta(seconds=leapseconds)
        self.datetime = gps_dt

    