"""The analysis of a planned campaign: a planner's run (GOSE) read and shaped for its plots.

A planner writes its plan as tables (CSV) in a run folder (``<run>/plan/``):
the schedule, the empty blocks, the coverage of the detectors, the impact on
the other telescopes. These modules read them (:mod:`.reader`) and turn them
into plot-ready tables -- one function per chart or table of the planner's
pages; the drawing is the interface's.

Modules
-------
reader
    A run's tables and its configuration (POLOCALC_ENG_PRE / _POST read as one project).
schedule
    The campaign day by day: the Gantt's flights, its empty blocks, a project's shared window.
fields
    The fields flown: what was flown where and when, on the site and in each telescope's sky.
coverage
    The wafers and detectors reached, how often, at which rotations, and for how long.
impact
    The other telescopes' scans the drone disturbs, for how long, and what would change it.
distance
    How close an observing telescope's boresight comes to the drone, flight by flight.
fieldsearch
    What a field search found: its fields, when each can be flown, the landscape, the Sun.
"""

from pils.analyze.plan.reader import PlanRun, project_name

__all__ = ["PlanRun", "project_name"]
