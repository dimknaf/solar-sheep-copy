"""envs/isaac_rover/solar_sheep_tasks/traverse/mdp/commands_cfg.py -- config of the rover's pose command.

Kept apart from commands.py the way Isaac Lab keeps commands_cfg.py apart from pose_2d_command.py:
the env config is imported before Kit starts, so it may only import config modules. The command class
itself is named by a lazy "{DIR}.commands:RoverPose2dCommand" string and imported once the sim runs.
"""

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.commands_cfg import TerrainBasedPose2dCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .commands import RoverPose2dCommand


@configclass
class RoverPose2dCommandCfg(TerrainBasedPose2dCommandCfg):
    """Go to (x, y) and end up facing ``heading``; a new target as soon as this one is reached.

    Inherits the heading-only ``Ranges`` of the terrain-based command. Set ``simple_heading=False``
    (random heading, as in the MJX prototype) and ``resampling_time_range`` to something huge: targets
    change on reach, not on a timer.
    """

    class_type: type["RoverPose2dCommand"] | str = "{DIR}.commands:RoverPose2dCommand"

    distance_range: tuple[float, float] = (0.8, 2.5)
    """Distance of a new target from the robot (not from the env origin) [m]."""

    max_target_radius: float = 3.4
    """Targets stay within this distance of the env origin [m] (0.85 x the 4 m bounds radius)."""

    reach_radius: float = 0.35
    """A target is reached within this xy distance [m] ...  (MJX REACH_R)"""

    reach_heading: float = 0.5
    """... and this absolute heading error [rad]. (MJX REACH_YAW)"""
