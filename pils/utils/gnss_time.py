"""GPS time: its epoch, the week, and the leap seconds between GPS time and UTC."""

import datetime

GPS_EPOCH = datetime.datetime(1980, 1, 6)
"""Start of GPS time (naive, 1980-01-06 00:00:00)."""

GPS_EPOCH_UNIX = 315964800.0
"""Start of GPS time, as a UNIX timestamp."""

SECONDS_PER_WEEK = 604800

# GPS - UTC (leap seconds), from the date (UTC) each value starts, newest first
GPS_UTC_LEAP_SECONDS = (
    ("2017-01-01", 18),
    ("2015-07-01", 17),
    ("2012-07-01", 16),
    ("2009-01-01", 15),
    ("2006-01-01", 14),
    ("1999-01-01", 13),
    ("1997-07-01", 12),
    ("1996-01-01", 11),
    ("1994-07-01", 10),
    ("1993-07-01", 9),
    ("1992-07-01", 8),
    ("1991-01-01", 7),
    ("1990-01-01", 6),
    ("1988-01-01", 5),
    ("1985-07-01", 4),
    ("1983-07-01", 3),
    ("1982-07-01", 2),
    ("1981-07-01", 1),
)


def leap_seconds(when: datetime.date | datetime.datetime) -> int:
    """GPS - UTC in seconds at a date or (naive) datetime.

    Examples
    --------
    >>> leap_seconds(datetime.date(2025, 12, 6))
    18
    >>> leap_seconds(datetime.datetime(2016, 12, 31, 23, 59))
    17
    """
    if isinstance(when, datetime.datetime):
        when = when.date()
    for start, leap in GPS_UTC_LEAP_SECONDS:
        if when >= datetime.date.fromisoformat(start):
            return leap
    return 0
