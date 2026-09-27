"""envs/isaac_rover/solar_sheep_tasks -- SolarSheep's Isaac Lab tasks, kept outside Isaac Lab's own tree.

Importing this package registers the tasks with Gym; envs/isaac_rover/train.py and play.py do that and
then hand over to Isaac Lab's unified RL command line. One sub-package per skill: traverse today; the
battery swap and the whole solar-sheep field join here later.
"""

from . import traverse  # noqa: F401
