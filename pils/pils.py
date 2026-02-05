from typing import Any, Dict, List, Optional

from pils.flight import Flight
from pils.loader.path import PathLoader
from pils.loader.stout import StoutLoader


class PILS:

    def __init__(
        self,
        use_stout: bool = True,
        base_path: str = "/mnt/data/POLOCALC",
        campaign_id: Optional[str] = None,
        campaign_name: Optional[str] = None,
        flight_id: Optional[str] = None,
        flight_name: Optional[str] = None,
    ):

        if use_stout:
            self.loader = StoutLoader()
        else:
            self.loader = PathLoader(base_path)

        self.__stout_flag = use_stout

        if campaign_id or campaign_name:

            self._tmp_flight = self.loader.load_all_campaign_flights(
                campaign_id=campaign_id, campaign_name=campaign_name
            )

        elif flight_id or flight_name:

            self._tmp_flight = self.loader.load_single_flight(
                flight_id=flight_id, flight_name=flight_name
            )

        self.flights = []

        if self._tmp_flight:
            for flight in self._tmp_flight:
                # Handle both string (flight ID) and dict (flight info) cases
                if isinstance(flight, str):
                    # If flight is a string, it might be flight ID or name
                    # Try to load the flight data
                    flight_data = self.loader.load_single_flight(
                        flight_id=flight if not flight.startswith("FLIGHT_") else None,
                        flight_name=flight if flight.startswith("FLIGHT_") else None,
                    )
                    if flight_data:
                        # flight_data might be a list with one element
                        if isinstance(flight_data, (list, tuple)) and len(flight_data) > 0:
                            flight_info: Dict[str, Any] = flight_data[0]  # type: ignore
                        elif isinstance(flight_data, dict):
                            flight_info = flight_data
                        else:
                            continue
                    else:
                        continue
                else:
                    # Assume it's already flight info dictionary
                    flight_info = flight

                tmp = Flight(flight_info)  # type: ignore

                self.flights.append(tmp)

    def load_drone_data(self, dji_dat_loader: bool = True, drone_model=None):

        for flight in self.flights:

            flight.add_drone_data(dji_dat_loader=dji_dat_loader, drone_model=drone_model)

    def load_sensor_data(self, sensor_name: List[str]):

        for flight in self.flights:

            flight.add_sensor_data(sensor_name=sensor_name)

    def load_all_data(self, dji_dat_loader: bool = True, drone_model=None):

        sensor_list = ["gps", "imu", "inclinometer", "adc"]

        self.load_drone_data(dji_dat_loader=dji_dat_loader, drone_model=drone_model)
        self.load_sensor_data(sensor_name=sensor_list)
