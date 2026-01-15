import os
import datetime
import pandas as pd

DATADIR = "/data/POLOCALC/campaigns"


def extract_timestamp_from_name(name):
    """
    Extract timestamp from YYYYMMDD_HHMMSS*
    """
    try:
        ts = datetime.datetime.strptime(name[:15], "%Y%m%d_%H%M%S")
        return pd.Timestamp(ts, tz="UTC")
    except Exception:
        return None


class PathHandler:
    """
    Path resolver for a single flight directory.

    Expected structure:
    flight_YYYYMMDD_HHMM/
        aux/
            YYYYMMDD_HHMMSS_file.log
            camera/*.mp4
            sensors/*.bin
        drone/
            YYYYMMDD_HHMMSS_drone.csv
            YYYYMMDD_HHMMSS_litchi.csv
    """

    def __init__(self, flight_dir):
        self.flight_dir = os.path.abspath(flight_dir)

        if not os.path.isdir(self.flight_dir):
            raise FileNotFoundError(
                f"[PathHandler] Flight directory not found: {self.flight_dir}"
            )

        self.aux_dir = os.path.join(self.flight_dir, "aux")
        self.drone_dir = os.path.join(self.flight_dir, "drone")

        self.drone = None
        self.litchi = None
        self.logfile = None
        self.camera = None

        self.gps = None
        self.adc = None
        self.inclino = None
        self.baro = None
        self.gyro = None
        self.accelero = None
        self.magneto = None
        self.LM76 = None

        self.timestamp = None

    def get_filenames(self):
        # --------------------------------------------------
        # Drone files
        # --------------------------------------------------
        drone_files = [
            f for f in os.listdir(self.drone_dir)
            if f.endswith("_drone.csv")
        ]
        if not drone_files:
            raise FileNotFoundError("[PathHandler] No drone CSV found")

        self.drone = os.path.join(self.drone_dir, drone_files[0])
        self.timestamp = extract_timestamp_from_name(drone_files[0])

        # Litchi
        litchi_files = [
            f for f in os.listdir(self.drone_dir)
            if f.endswith("_litchi.csv")
        ]
        self.litchi = (
            os.path.join(self.drone_dir, litchi_files[0])
            if litchi_files else None
        )

        # --------------------------------------------------
        # Aux files
        # --------------------------------------------------
        aux_files = os.listdir(self.aux_dir)

        # logfile
        logfiles = [f for f in aux_files if f.endswith("_file.log")]
        if not logfiles:
            raise FileNotFoundError(f"[PathHandler] No _file.log found in {self.aux_dir}")
        self.logfile = os.path.join(self.aux_dir, logfiles[0])

        # camera
        camera_dir = os.path.join(self.aux_dir, "camera")
        if os.path.isdir(camera_dir):
            for f in os.listdir(camera_dir):
                if f.lower().endswith(".mp4"):
                    self.camera = os.path.join(camera_dir, f)
                    break

        # --------------------------------------------------
        # Sensors
        # --------------------------------------------------
        sensors_dir = os.path.join(self.aux_dir, "sensors")
        if os.path.isdir(sensors_dir):
            for f in os.listdir(sensors_dir):
                if f.endswith("_GPS.bin"):
                    self.gps = os.path.join(sensors_dir, f)
                elif f.endswith("_ADC.bin"):
                    self.adc = os.path.join(sensors_dir, f)
                #elif f.endswith("_INC_ins.csv"):
                    #self.inclino = os.path.join(sensors_dir, f)
                elif f.endswith("_BAR.bin"):
                    self.baro = os.path.join(sensors_dir, f)
                elif f.endswith("_GYR.bin"):
                    self.gyro = os.path.join(sensors_dir, f)
                elif f.endswith("_ACC.bin"):
                    self.accelero = os.path.join(sensors_dir, f)
                elif f.endswith("_MAG.bin"):
                    self.magneto = os.path.join(sensors_dir, f)
                elif f.endswith(".csv") and "TMP" in f:
                    self.LM76 = os.path.join(sensors_dir, f)
