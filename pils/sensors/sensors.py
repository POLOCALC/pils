from pils.sensors.adc import ADC
from pils.sensors.gps import GPS
from pils.sensors.IMU import IMU
from pils.sensors.inclinometer import Inclinometer
from pils.sensors.LM76 import LM76

sensor_config = {
    "gps": {
        "class": GPS,
        "load_method": "load_data",
    },
    "imu": {
        "class": IMU,
        "load_method": "load_all",
    },
    "adc": {"class": ADC, 
            "load_method": "load_data"
    },
    "inclinometer": {
        "class": Inclinometer,
        "load_method": "load_data",
    },
    "lm76": {"class": LM76, 
             "load_method": "load_data"
    },
}
