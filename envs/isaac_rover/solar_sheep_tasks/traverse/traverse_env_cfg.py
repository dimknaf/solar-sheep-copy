"""envs/isaac_rover/solar_sheep_tasks/traverse/traverse_env_cfg.py -- the traverse task, manager-based.

Port of envs/traverse/train.py (MJX + Brax) onto Isaac Lab + PhysX: "drive to (x, y), end up pointing
at yaw, do not hit anything, do not tip", targets resampled on reach. Same 32-float observation in the
same order and scale, same 4 wheel-speed actions, same reward terms and relative weights (REWARD_W),
same terminations. What changed with the engine is listed where it happens:
    slopes, rubble and obstacles are real terrain tiles (terrain_cfg.py), not a body force and mocap
    boxes; the curriculum is Isaac Lab's terrain-row curriculum driven by targets reached;
    collisions come from a PhysX contact sensor on the chassis, not a geometric overlap test.

    TraverseRoughEnvCfg   SolarSheep-Traverse-Rough-Rover   terrain generator + obstacles + curriculum
    TraverseFlatEnvCfg    SolarSheep-Traverse-Flat-Rover    plane, pose tracking only (the escape hatch)
Both keep all 32 observations (the fan just reads clear on the plane), so checkpoints move between them.

Physics: PhysX at 240 Hz, 5 substeps per policy step (48 Hz), 25 s episodes. Ground mu 1.0 with the
"multiply" combine mode, so the tyre-ground pair takes the tyre's 0.65 (as in measure_rover.py).

Per-episode metrics in TensorBoard:
    targets reached   Metrics/pose_command/targets_reached   (success count; drives the curriculum)
    final distance    Metrics/pose_command/error_pos, Metrics/pose_command/error_heading
    collisions        Episode_Reward/collide = -(seconds in contact) / 25 s
    tip-overs         Episode_Termination/tipped
    difficulty        Curriculum/terrain_levels (rough task)

Viewing and safety (owner rule): a Viser live view bound to 127.0.0.1 only (reached through the SSH
tunnel, scripts/gpu/watch.sh), share=False, and a headless Kit viewport that follows env 0's rover
(it moves between terrain rows with the curriculum). Nothing listens beyond localhost; nothing is
uploaded. Clips: only with --video on train.py / play.py; Isaac Lab's entrypoint then records from the
Kit viewport into <run>/videos/{train,play} (train.py sets the clip length and interval). No recorder
is configured here: env_cfg.video_recorders is built and stepped on every run, --video or not.

Top-level imports are config modules only: this file is read before Kit starts.
"""

import math

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.viser import ViserVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp
from .rover_cfg import ROVER_CFG, WHEEL_JOINTS
from .terrain_cfg import ROVER_TERRAINS_CFG, TARGET_EXTENT

# --- the prototype's constants (envs/traverse/train.py) ----------------------------------------------
SLOPE_MAX_DEG = 12.0          # normaliser of the slope observation
RF_MAX = 3.0                  # rangefinder range [m]; observation is range / RF_MAX, 1.0 = clear
RF_POS = (0.34, 0.0, 0.005)   # chassis frame: the sensor pod, same origin as the MJX sites (~0.18 m up)
TIP_TERM = 0.40               # terminate when body up-vector z < 0.40
TIP_SOFT = 0.90               # tilt penalty below this
BOUNDS_R = 4.0                # [m] from the env origin before the bounds penalty (half a tile)
# Rangefinder tilt, decided here: +4 deg up, not level. On the rough tiles (3 cm bumps under a 0.22 m
# wheelbase) the chassis pitches by up to ~8 deg; a level ray 0.18 m up then strikes flat ground at
# 0.18 / tan(8 deg) = 1.3 m and reads as an obstacle that is not there. Tilted 4 deg up, the ground stays
# beyond 3 m for pitch-down up to 7.4 deg, and the ray tops out at 0.18 + 3 tan(4 deg) = 0.39 m, still
# under the 0.5 m obstacles, so every obstacle in range is still seen. Approaching the foot of the
# steepest (9.6 deg) ramp, a level fan reads it as a wall from 1.9 m out; the tilted one only from 0.7 m.
RF_TILT_DEG = 4.0
COMMAND = "pose_command"


def _wheels() -> SceneEntityCfg:
    """The four wheel joints in action order (fl, fr, rl, rr)."""
    return SceneEntityCfg("robot", joint_names=WHEEL_JOINTS, preserve_order=True)


def _ground_material() -> PhysxRigidBodyMaterialCfg:
    return PhysxRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, friction_combine_mode="multiply",
                                     restitution_combine_mode="multiply")


##
# Scene
##


@configclass
class TraverseSceneCfg(InteractiveSceneCfg):
    """Rover on generated terrain, a 9-ray rangefinder fan and a chassis contact sensor."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROVER_TERRAINS_CFG,
        max_init_terrain_level=2,              # start in the easiest rows, like the prototype's u = 0
        collision_group=-1,
        physics_material=_ground_material(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.33, 0.31, 0.28)),   # local, no download
        debug_vis=False,
    )
    robot: ArticulationCfg = ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # 9 rays, -90..+90 deg every 22.5 deg (ray 0 = right, +y = left), 3 m, from the sensor pod. PhysX ray
    # casting takes ONE static mesh: /World/ground, which is why obstacles are part of the terrain.
    rangefinders = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/chassis",
        offset=RayCasterCfg.OffsetCfg(pos=RF_POS),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, vertical_fov_range=(RF_TILT_DEG, RF_TILT_DEG), horizontal_fov_range=(-90.0, 90.0),
            horizontal_res=22.5),
        max_distance=RF_MAX,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    # Chassis (and panel) contacts only: wheel-ground contact is normal driving, not a collision.
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/Geometry/chassis", history_length=3)
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=2500.0))


##
# MDP
##


@configclass
class CommandsCfg:
    pose_command = mdp.RoverPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,                       # random target heading, as in the prototype
        resampling_time_range=(1.0e6, 1.0e6),       # resample on reach (mdp/commands.py), never on a timer
        debug_vis=True,
        distance_range=(0.8, 2.5),
        max_target_radius=TARGET_EXTENT,
        reach_radius=0.35,
        reach_heading=0.5,
        ranges=mdp.RoverPose2dCommandCfg.Ranges(heading=(-math.pi, math.pi)),
    )


@configclass
class ActionsCfg:
    # one command per wheel in [-1, 1] -> +-pi rad/s (rover.xml ctrlrange), fl, fr, rl, rr
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=WHEEL_JOINTS, preserve_order=True, scale=math.pi,
        use_default_offset=False, clip={"wheel_.*": (-math.pi, math.pi)})


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """observe() of envs/traverse/train.py, same order and scale: 32 floats."""

        # 3  chassis linear velocity (body) / 0.5
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        # 3  chassis angular velocity (body) / 1
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # 3  up-vector in the body frame = -projected gravity
        up_vector = ObsTerm(func=mdp.projected_gravity, scale=-1.0)
        # 2  slope: downhill direction (body) / sin 12 deg
        slope = ObsTerm(func=mdp.slope_b, params={"max_slope_deg": SLOPE_MAX_DEG})
        # 4  wheel speeds / pi (fl, fr, rl, rr)
        wheel_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": _wheels()}, scale=1.0 / math.pi)
        # 2  target dx, dy (heading frame) / 2 m, clipped +-2
        target_offset = ObsTerm(func=mdp.target_offset_b, params={"command_name": COMMAND})
        # 2  sin, cos of the heading error
        heading_error = ObsTerm(func=mdp.heading_error_sin_cos, params={"command_name": COMMAND})
        # 9  rangefinder fan / 3 m, 1.0 = clear
        rangefinders = ObsTerm(
            func=mdp.rangefinders, params={"sensor_cfg": SceneEntityCfg("rangefinders"), "max_distance": RF_MAX})
        # 4  previous action (keeps the action-rate penalty Markovian)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False      # the prototype had no observation noise
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4), "z": (0.01, 0.03), "roll": (-0.05, 0.05),
                           "pitch": (-0.05, 0.05), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    reset_wheels = EventTerm(      # wheels back to their default: 0 rad, 0 rad/s
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    # the prototype's random kicks: 0.002-0.008 per 20 ms step = one every 2.5-10 s
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.5, 10.0),
        params={"velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """REWARD_W of envs/traverse/train.py, term for term, except pos + head (potential form, see shaping).

    The manager multiplies by weight * step_dt.
    """

    progress = RewTerm(func=mdp.progress_to_target, weight=2.0, params={"command_name": COMMAND, "max_speed": 0.5})
    # pos (1.0, std 0.6) + head (1.5, std 0.8) of the prototype, paid as a potential DIFFERENCE so that
    # hovering next to the target earns nothing (the per-step form made stopping short the optimum).
    shaping = RewTerm(
        func=mdp.target_potential, weight=1.0,
        params={"command_name": COMMAND, "gamma": 0.995, "w_pos": 1.0, "std_pos": 0.6, "w_head": 1.5, "std_head": 0.8})
    reach = RewTerm(func=mdp.target_reached, weight=20.0, params={"command_name": COMMAND})
    alive = RewTerm(func=mdp.is_alive, weight=0.05)
    # the prototype averages over the 4 wheels; action_l2 / action_rate_l2 sum them, hence / 4
    ctrl = RewTerm(func=mdp.action_l2, weight=-0.02 / 4)
    rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05 / 4)
    energy = RewTerm(func=mdp.wheel_power, weight=-0.01, params={"asset_cfg": _wheels()})
    tip = RewTerm(func=mdp.tilt, weight=-4.0, params={"soft_limit": TIP_SOFT})
    tipover = RewTerm(func=mdp.is_terminated_term, weight=-10.0, params={"term_keys": "tipped"})
    spin = RewTerm(func=mdp.spin, weight=-0.5, params={"max_rate": 0.5})
    # per step with the chassis in contact (> 1 N). The prototype's collide_depth (-2 x overlap depth)
    # has no contact-sensor equivalent and is left out; collide stays below progress + pos as before.
    collide = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="chassis"), "threshold": 1.0},
    )
    idle = RewTerm(
        func=mdp.idle_far_from_target, weight=-0.3,
        params={"command_name": COMMAND, "min_speed": 0.05, "min_distance": 0.8})
    bounds = RewTerm(func=mdp.out_of_bounds, weight=-1.0, params={"radius": BOUNDS_R})


@configclass
class TerminationsCfg:
    """Time limit, tip-over (up-vector z < 0.40), non-finite state. No collision termination, by design."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    tipped = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.acos(TIP_TERM)})
    nonfinite = DoneTerm(func=mdp.root_state_nonfinite)


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_nav, params={"command_name": COMMAND, "promote_at": 2, "demote_at": 0})


##
# Environments
##


@configclass
class TraverseRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Pose + heading tracking with obstacle avoidance over rover-scale rough terrain."""

    # PhysX, explicitly: no preset system here (Isaac Lab's velocity env defaults to Newton; we do not).
    # Contact-patch budget as in Isaac Lab's rough velocity env (thousands of robots on a mesh terrain).
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 240.0, render_interval=5, physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    )
    scene: TraverseSceneCfg = TraverseSceneCfg(num_envs=4096, env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 5                         # 240 Hz physics, 48 Hz policy (the prototype: 50 Hz)
        self.episode_length_s = 25.0
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.rangefinders.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        # viewing -- owner rule: bind 127.0.0.1 only, no share link. The SSH tunnel is the only way in.
        self.sim.visualizer_cfgs = [
            ViserVisualizerCfg(bind_address="127.0.0.1", port=8080, open_browser=False, share=False,
                               max_visible_envs=64),
            # follows env 0's rover each step ("env" would fix the camera on the tile env 0 started on;
            # the curriculum moves it). eye/lookat are offsets from the rover root, 0.18 m up.
            KitVisualizerCfg(headless=True, origin_type="asset", origin_track_path="robot", origin_env_index=0,
                             eye=(3.0, 3.0, 2.0), lookat=(0.0, 0.0, 0.0), window_width=1280, window_height=720),
        ]

    def play_mode(self):
        """Playback: Isaac Lab's defaults (<= 50 envs), every terrain row, no pushes."""
        super().play_mode()
        self.events.push_robot = None
        generator = self.scene.terrain.terrain_generator
        if generator is not None:
            self.scene.terrain.max_init_terrain_level = None
            generator.num_rows = 5
            generator.num_cols = 5
            generator.curriculum = False


@configclass
class TraverseFlatEnvCfg(TraverseRoughEnvCfg):
    """Pose + heading tracking on a plane: the prototype's --rough 0 --obstacles 0 escape hatch."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        # fewer kicks: at difficulty 0 the prototype pushed at its slowest rate, one every ~10 s
        self.events.push_robot.interval_range_s = (8.0, 12.0)
