from typing import List, Dict, Optional, Any
from pils.loader.stout import StoutLoader
from pils.loader.path import PathLoader
from pils.flight import Flight


class PILS:

    def __init__(
        self,
        use_stout: bool = True,
        campaign_id: Optional[str] = None,
        campaign_name: Optional[str] = None,
        flight_id: Optional[str] = None,
        flight_name: Optional[str] = None,
    ):

        if use_stout:
            self.loader = StoutLoader()
        else:
            self.loader = PathLoader()


        if campaign_id or campaign_name:

            self._tmp_flight = self.loader.load_all_campaign_flights(
                campaign_id=campaign_id, campaign_name=campaign_name
            )

        self.flights = []

        for flight in self._tmp_flight:

            tmp = Flight(flight)

            self.flights.append(tmp)

    def load_drone_data(self, dji_drone_loader: str = "dat"):

        for flight in self.flights:

            flight.load_drone_data(dji_drone_loader=dji_drone_loader)


    def load_sensor_data(self, sensor_name: List[str]):

        for flight in self.flights:

            flight.load_sensor_data(sensor_name=sensor_name)

    def load_all_data(self, dji_drone_loader: str = "dat"):

        sensor_list = ["gps", "imu", "inclino", "adc"]

        self.load_drone_data(dji_drone_loader=dji_drone_loader)
        self.load_sensor_data(sensor_name=sensor_list)