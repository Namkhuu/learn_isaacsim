from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab import cloner

from gripper import (
    GRIPPER_START_BODY_NAME,
    GRIPPER_JOINT_NAMES,
    GRIPPER_CLOSE_DISTANCE,
)


# Shared physics parameters (single source of truth)
base_static_friction = 4.0
base_dynamic_friction = 4.0
base_restitution = 0.0

# Domain randomization ranges
mass_scale_range = (0.85, 1.15)


@configclass
class EventCfg:
    """Domain randomization events."""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (
                base_static_friction * 0.5,
                base_static_friction * 3,
            ),
            "dynamic_friction_range": (
                base_dynamic_friction * 0.5,
                base_dynamic_friction * 3,
            ),
            "restitution_range": (base_restitution, base_restitution),
            "num_buckets": 64,
        },
    )
    add_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": mass_scale_range,
            "operation": "scale",
        },
    )


@configclass
class MyRobotEnvCfg(DirectRLEnvCfg):
    episode_length_s = 12
    decimation = 4
    action_scale = 0.5  # max joint travel from default pose: 0.5 rad
    action_smoothing = 0.25  # fraction of new action applied each step; lower = smoother
    action_space = 8
    num_objects_in_scene = 1
    # +1 dist, +4 cube quat, +gripper joint pos (see _get_observations)
    observation_space = 9 + (action_space * 3) + 3 * num_objects_in_scene + 1 + 4 + len(GRIPPER_JOINT_NAMES)
    state_space = 0
    events: EventCfg = EventCfg()

    sim: SimulationCfg = SimulationCfg(
        dt=1/200,
        render_interval=decimation,  # one viewport frame per control step (~50 Hz ≈ realtime)
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=base_static_friction,
            dynamic_friction=base_dynamic_friction,
            restitution=base_restitution,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=base_static_friction,
            dynamic_friction=base_dynamic_friction,
            restitution=base_restitution,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1000, env_spacing=2.0, replicate_physics=True)

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_[^/]+/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).resolve().parent / "robot_3.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.2,              # linear drag coefficient (1/s)
                angular_damping=0.2,             # angular drag coefficient (1/s)
                max_linear_velocity=10.0,        # body speed cap: 10 m/s
                max_angular_velocity=20.0,       # body spin cap: 20 rad/s
                max_depenetration_velocity=1.0,  # contact correction cap: 1 m/s
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.01),
            joint_pos={".*": 0.0},
        ),
        # Actuator defaults assume revolute joints. For prismatic, use N, m/s, N/m, N·s/m, kg.
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=3.0,    # joint motor torque: 3 N·m
                velocity_limit_sim=8.0,  # max joint speed: 8 rad/s
                stiffness=20.0,          # PD stiffness: 20 N·m/rad
                damping=4.0,             # PD damping: 4 N·m·s/rad
                armature=0.01,           # rotor inertia at joint: 0.01 kg·m²
            ),
        },
    )

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_[^/]+/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0, 0.30, 0.025),
        ),
    )
    


class MyRobotEnv(DirectRLEnv):
    cfg: MyRobotEnvCfg

    def __init__(self, 
                cfg: MyRobotEnvCfg, 
                render_mode: str | None = None, 
                **kwargs
            ):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, 
                gym.spaces.flatdim(self.single_action_space), 
                device=self.device)
                
        self._smoothed_actions = torch.zeros_like(self._actions)

        if GRIPPER_START_BODY_NAME not in self._robot.data.body_names:
            raise ValueError(
                f"GRIPPER_START_BODY_NAME '{GRIPPER_START_BODY_NAME}' not found in robot body_names. "
                "Set it in gripper.py to a valid body name from the startup printout."
            )
        self._gripper_start_body_index = self._robot.data.body_names.index(GRIPPER_START_BODY_NAME)
        self._gripper_body_indices = list(range(self._gripper_start_body_index, len(self._robot.data.body_names)))
        # Align using the link just before the gripper start when available.
        self._align_body_idx = max(self._gripper_start_body_index - 1, self._gripper_start_body_index)

        self._gripper_joint_indices = [
            self._robot.data.joint_names.index(name)
            for name in GRIPPER_JOINT_NAMES
            if name in self._robot.data.joint_names
        ]
        if len(self._gripper_joint_indices) == 0:
            raise ValueError(
                "No GRIPPER_JOINT_NAMES were found in robot joint_names. "
                "Set GRIPPER_JOINT_NAMES in gripper.py to valid joint names."
            )

        limits = self._robot.data.soft_joint_pos_limits
        if limits.dim() == 3:
            low = limits[0, :, 0]
            high = limits[0, :, 1]
        else:
            low, high = limits[:, 0], limits[:, 1]
        gripper_low = low[self._gripper_joint_indices]
        gripper_high = high[self._gripper_joint_indices]
        self._gripper_open_value = gripper_low.unsqueeze(0).to(self.device)
        self._gripper_close_value = gripper_high.unsqueeze(0).to(self.device)

        self._cube_start_z = torch.zeros(self.num_envs, device=self.device)

        print("\n=== Robot Joints ===")
        print(f"Joint names:    {self._robot.joint_names}")
        print(f"Num joints:     {self._robot.num_joints}")
        print(f"Body names:     {self._robot.body_names}")
        print(f"Gripper start body: {GRIPPER_START_BODY_NAME} (index {self._gripper_start_body_index})")
        print(f"Align body index:   {self._align_body_idx} ({self._robot.body_names[self._align_body_idx]})")
        print(f"Gripper joints: {GRIPPER_JOINT_NAMES}")
        print(f"Gripper bodies (derived): {[self._robot.body_names[i] for i in self._gripper_body_indices]}")
        print("===================\n")

    def _setup_scene(self):
        from isaaclab import cloner
            # Create assets
        self._robot = Articulation(self.cfg.robot)
        self._cube = RigidObject(self.cfg.cube)
                # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
                # Generate environment positions
        positions = cloner.grid_transforms(
            self.scene.num_envs,
            self.scene.cfg.env_spacing,
        )[0]
                # Build clone plan using YOUR installed API
        plan = cloner.clone_plan_from_env_0(
            self.scene.cfg.clone_cfg,
            [self.cfg.robot, self.cfg.cube],
            self.scene.num_envs,
            self.scene.cfg.env_spacing,
            positions=positions,
        )
                # Replicate
        cloner.replicate(
            plan,
            replicate_physics=self.scene.cfg.replicate_physics,
        )
                # Register assets with scene
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["cube"] = self._cube
        print("SCENE KEYS:", list(self.scene.keys()))
                # Collision filtering
        if self.device == "cpu":
            self.scene.filter_collisions(
                global_prim_paths=[self.cfg.terrain.prim_path]
            )
                # Light
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().view(self.num_envs, -1)
        alpha = self.cfg.action_smoothing
        self._smoothed_actions = (1.0 - alpha) * self._smoothed_actions + alpha * self._actions
        self._processed_actions = self.cfg.action_scale * self._smoothed_actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        gripper_pos_w = self._robot.data.body_pos_w[:, self._gripper_start_body_index, :]
        cube_pose = self._cube.data.root_link_pose_w.torch
        to_cube_w = cube_pose[:, :3] - gripper_pos_w
        root_quat_inv = self._robot.data.root_link_pose_w.torch[:, 3:7] * torch.tensor([-1, -1, -1, 1], device=self.device)
        to_cube_b = quat_apply(root_quat_inv, to_cube_w)
        dist_to_cube = torch.norm(to_cube_w, dim=1, keepdim=True)
        cube_quat = cube_pose[:, 3:7]
        gripper_joint_pos = self._robot.data.joint_pos[:, self._gripper_joint_indices]

        obs = torch.cat([
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self._robot.data.joint_vel,
            self._actions,
            to_cube_b,
            dist_to_cube,
            cube_quat,
            gripper_joint_pos,
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        gripper_pos = self._robot.data.body_pos_w[:, self._gripper_start_body_index, :]
        obj_pos = self._cube.data.root_link_pose_w.torch[:, :3]
        dist = torch.norm(obj_pos - gripper_pos, dim=1)

        # ------------------------------------------------------------------
        # REACH — penalise distance directly, no gating, no saturation.
        # The robot always loses reward for being far away.
        # ------------------------------------------------------------------
        reach_reward = -dist

        # ------------------------------------------------------------------
        # GRASP — reward finger closure directly, completely independent of
        # position. 0 = fully open, 1 = fully closed.
        # No proximity gate: the robot always benefits from closing its fingers.
        # The only way to score on both reach AND grasp is to be near the cube
        # with fingers closed — i.e. actually grasp it.
        # ------------------------------------------------------------------
        gripper_joint_pos = self._robot.data.joint_pos[:, self._gripper_joint_indices]
        gripper_range = self._gripper_close_value - self._gripper_open_value
        gripper_range = torch.clamp(gripper_range, min=1e-4)
        close_frac = torch.mean(
            torch.clamp(
                (gripper_joint_pos - self._gripper_open_value.expand(self.num_envs, -1)) / gripper_range,
                0.0, 1.0,
            ),
            dim=1,
        )  # 0 = fully open, 1 = fully closed

        # ------------------------------------------------------------------
        # LIFT — reward cube height directly, no gating.
        # Any upward cube movement scores, whether from a grasp or a nudge.
        # Combined with reach and grasp rewards the only way to score on all
        # three simultaneously is to actually pick the cube up.
        # ------------------------------------------------------------------
        lift_height = torch.clamp(obj_pos[:, 2] - self._cube_start_z, min=0.0)

        # ------------------------------------------------------------------
        # Penalty — small joint velocity penalty to keep motion smooth.
        # ------------------------------------------------------------------
        joint_vel_penalty = torch.sum(self._robot.data.joint_vel ** 2, dim=1)

        reward = (
            4.0  * reach_reward       # always penalising distance
            + 3.0  * close_frac       # always rewarding finger closure
            + 8.0 * lift_height      # big reward for cube going up
            - 0.005 * joint_vel_penalty
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fallen = self._robot.data.root_link_pose_w.torch[:, 2] < -0.1
        return fallen, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            # Asset internals use Warp indices in Isaac Lab 3; RL buffers use Torch.
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._smoothed_actions[env_ids] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        cube_state = self._cube.data.default_root_state[env_ids].clone()
        cube_state[:, :3] += self._terrain.env_origins[env_ids]
        cube_state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        cube_state[:, 7:] = 0.0

        self._cube.write_root_pose_to_sim(cube_state[:, :7], env_ids)
        self._cube.reset(env_ids)
        self._cube_start_z[env_ids] = cube_state[:, 2]
