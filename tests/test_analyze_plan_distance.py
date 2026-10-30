"""An observing telescope's boresight against the drone, flight by flight (pils.analyze.plan.distance)."""

from pils.analyze.plan import distance as D

ROWS = [
    {"date": "2026-11-27", "start": "09:00", "worst_deg": 12.0, "median_deg": 14.0, "best_deg": 16.0, "static_deg": 9.0},
    {"date": "2026-11-27", "start": "09:30", "worst_deg": 8.0, "median_deg": 9.0, "best_deg": 11.0, "static_deg": 12.0},
    {"date": "2026-11-28", "start": "09:00", "worst_deg": None, "median_deg": None, "best_deg": None, "static_deg": None},
]


def test_the_flights_followed_and_their_counts():
    assert [r["start"] for r in D.followed(ROWS)] == ["09:30", "09:00"]
    assert D.summary(ROWS, 10.0) == {"flights": 2, "worst_within": 1, "median_within": 1, "closest_deg": 8.0,
                                     "static_within": 1}


def test_the_median_case_histogram():
    h = D.median_histogram(ROWS, bins=5)
    assert h["median"] == 11.5 and sum(h["counts"]) == 2 and h["counts"][0] == 1 and h["counts"][-1] == 1
    assert D.median_histogram([]) == {"edges": [], "counts": [], "median": None}
