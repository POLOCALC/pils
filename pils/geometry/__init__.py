"""Geometry for the analysis of planned and flown flights alike.

The same functions place the drone in a telescope's focal plane and count the
detectors it crosses, whether its track was simulated by a planner (GOSE) or
measured (a flight's ``azel/<TELESCOPE>`` and the telescope's own pointing).
so3g, which some of them need, is imported only when they are called, so that
importing pils stays light.

Modules
-------
site
    Positions on the site (the telescopes, the drone) and how one sees the other.
focal_plane
    A telescope's wafers, and where they lie under a boresight rotation.
detectors
    A telescope's detectors: positions, bands and beams (the SO layouts shipped).
detector_tables
    A focal plane read from a detector table (CLASS, the LAT) instead.
crossings
    The drone in the focal plane, and the detectors it crosses.
"""
