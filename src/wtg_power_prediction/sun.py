import datetime as dt

import ephem


class SunPosition:
    def __init__(self, *, latitude: float, longitude: float) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self._observer = self._create_ephem_observer()
        self._sun = ephem.Sun()

    def _create_ephem_observer(self) -> ephem.Observer:
        observer = ephem.Observer()
        observer.lat = str(self.latitude)
        observer.lon = str(self.longitude)
        return observer

    def altitude(self, *, timestamp_utc: dt.datetime) -> float:
        self._observer.date = timestamp_utc
        self._sun.compute(self._observer)
        return self._sun.alt
