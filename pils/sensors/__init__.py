"""
Sensor modules for payload instruments.

Includes GPS, ADC, IMU (accelerometer, gyroscope, magnetometer, barometer),
camera, and kernel/inclinometer data processing.
"""

from pils.sensors.gps import GPS
from pils.sensors.adc import ADC
from pils.sensors.IMU import IMUSensor
from pils.sensors.camera import Camera
from pils.sensors.kernel import Inclinometer, decode_inclino

__all__ = [
    "GPS",
    "ADC",
    "IMUSensor",
    "Camera",
    "Inclinometer",
    "decode_inclino",
]
