"""envs/isaac_rover/solar_sheep_tasks/traverse/rover_cfg.py -- the rover as an Isaac Lab articulation.

The USD is the official-importer output of robot/rover.xml (robot/usd/convert_rover.py); to use another
copy pass --usd PATH to train.py / play.py, which set ROVER_USD before this module is imported
(scripts/gpu/isaac.sh does not forward the host's environment into the container). Its default prim
/solar_rover lands at {ENV_REGEX_NS}/Robot, so the articulation root (the chassis) is
{ENV_REGEX_NS}/Robot/Geometry/chassis and the four wheel joints are
wheel_fl, wheel_fr, wheel_rl, wheel_rr (revolute about +y; +velocity rolls the rover along +x).
Left side = fl, rl (+y); right side = fr, rr.

The wheel actuators are copied exactly from envs/isaac_rover/measure_rover.py, where they passed the
G2 parity gate against robot/SPEC.md (0.279 m/s at pi rad/s, 0.571 rad/s spin, climbs 14 deg):
    velocity servo kv 18 N*m*s/rad -> damping 18, stiffness 0;  forcerange +-2.4 N*m -> effort 2.4
    armature 0.01;  frictionloss 0.60 N*m -> friction 0.60;  joint damping 0.02 -> viscous 0.02
Variant "Physics" = "physx" keeps the PhysX drives (the "mujoco" variant adds uncommanded MuJoCo servos).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ROVER_USD = os.environ.get("ROVER_USD", "/data/runs/usd/rover_train/rover_train/rover_train.usda")
CHASSIS_Z = 0.175                                          # chassis origin above ground, wheels touching
WHEEL_JOINTS = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]   # action / observation order

ROVER_CFG = ArticulationCfg(
    # activate_contact_sensors: PhysX only reports contacts (ContactSensor on the chassis) for bodies
    # carrying the contact-report API, which this flag applies to every rigid body of the rover.
    spawn=sim_utils.UsdFileCfg(usd_path=ROVER_USD, variants={"Physics": "physx"}, activate_contact_sensors=True),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, CHASSIS_Z + 0.005), joint_pos={".*": 0.0},
                                               joint_vel={".*": 0.0}),
    actuators={"wheels": ImplicitActuatorCfg(
        joint_names_expr=["wheel_.*"], stiffness=0.0, damping=18.0, joint_effort_limit=2.4,
        armature=0.01, friction=0.60, dynamic_friction=0.60, viscous_friction=0.02)},
)
"""The rover; the scene sets ``prim_path`` with ``ROVER_CFG.replace(prim_path=...)``."""
