from pathhandler import PathHandler, DATADIR
from sensors.inclinometer import Inclinometer
from sensors.gps import GPS
from sensors.adc import ADC
from sensors.IMU import IMUSensor
from drones.litchi import Litchi
from drones.DJIDrone import DJIDrone
from drones.BlackSquareDrone import BlackSquareDrone
from sensors.camera import Camera

def drone_init(drone_model, drone_path):
    drone_model = drone_model.lower()
    if drone_model == "dji":
        return DJIDrone(drone_path)
    elif drone_model == "blacksquare":
        return BlackSquareDrone(drone_path)
    else:
        raise ValueError(f"[drone_init] Drone model '{drone_model}' is unknown. Should be 'DJI' or 'BlackSquare'.")


class Payload:
    def __init__(self, pathhandler):
        self.gps = GPS(pathhandler.gps)
        self.adc = ADC(pathhandler.adc)
        self.inclino = Inclinometer(pathhandler.inclino)
        self.baro = IMUSensor(pathhandler.baro, "baro")
        self.accelero = IMUSensor(pathhandler.accelero, "accelero")
        self.magneto = IMUSensor(pathhandler.magneto, "magneto")
        self.gyro = IMUSensor(pathhandler.gyro, "gyro")

    def load_data(self):
        for attr_name in vars(self):
            if attr_name.startswith("__"):
                continue

            attr = getattr(self, attr_name)
            # Call load_data() if the attribute has it
            if hasattr(attr, "load_data") and callable(getattr(attr, "load_data")) and getattr(attr, "path") is not None:
                attr.load_data()

class DataHandler:
    """
    Coordinates data loading from drone logs, payload sensors, and Litchi flight logs.
    Supports DJI and BlackSquare drone models.
    """
    def __init__(self, num, dirpath=DATADIR, drone_model="DJI"):

        #Load the paths handler and fetch all the sensors filenames
        self.paths = PathHandler(num=num, dirpath=dirpath)
        self.paths.get_filenames()

        if self.paths.logfile is None:
            print("[DataHandler] No matching logfile found. Stopped the process.")
            return

        #Define the drone model and initialize the drone object with the correct function
        self.drone_model = drone_model.lower()
        self.drone = drone_init(self.drone_model, self.paths.drone)

        #Initialize the payload object
        self.payload = Payload(self.paths)

        #Initialize the litchi object
        self.litchi = Litchi(self.paths.litchi)

        #Initialize the camera object
        self.camera = Camera(self.paths.camera)

    def load_data(self):
        for attr_name in vars(self):
            if attr_name.startswith("__"):
                continue

            attr = getattr(self, attr_name)

            # Call load_data() if the attribute has it
            if hasattr(attr, "load_data") and callable(getattr(attr, "load_data")):
                if hasattr(attr, "path") and getattr(attr, "path") is not None:
                    attr.load_data()
                elif not hasattr(attr, "path"):
                    attr.load_data()
                else:
                    print(f"[load_data] Could not load data for '{attr_name}'")
                    continue

