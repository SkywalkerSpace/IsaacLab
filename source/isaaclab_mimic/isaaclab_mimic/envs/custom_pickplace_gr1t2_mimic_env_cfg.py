# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mimic configuration for the local GR1T2 pick and place assets."""

from isaaclab.envs.mimic_env_cfg import (
    MimicEnvCfg,
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.pick_place.custom_pick_place_gr1t2_env_cfg import (
    CollectSceneCfg,
    CollectPickPlaceGR1T2EnvCfg,
)
from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import PickPlaceGR1T2ObservationsCfg
from isaaclab_tasks.contrib.robot_pov_camera_cfg import robot_pov_camera_cfg


@configclass
class CustomPickPlaceGR1T2MimicSceneCfg(CollectSceneCfg):
    """Custom pick-place scene with cameras mounted above both wrists."""

    robot_pov_cam = robot_pov_camera_cfg(
        parent_prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset_pos=(0.11999996, -0.00000233, 0.74674994),
        offset_rot=(-0.69303199, 0.69304552, -0.14034840, 0.14034565),
    )
    robot_pov_cam.height = 256
    robot_pov_cam.width = 256

    left_wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_pitch_link/LeftWristCam",
        update_period=0.0,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2.0)),
        # Mirror the right camera across the palm: mount on +Y and aim toward the fingers (-Z).
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.15, 0.0), rot=(0.92387953, 0.0, 0.0, 0.38268343), convention="ros"
        ),
    )
    right_wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_pitch_link/RightWristCam",
        update_period=0.0,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.15, 0.0), rot=(-0.92387953, 0.0, 0.0, 0.38268343), convention="ros"
        ),
    )


@configclass
class CustomPickPlaceGR1T2MimicObservationsCfg(PickPlaceGR1T2ObservationsCfg):
    """RGB observations from the head and both wrist cameras."""

    @configclass
    class PolicyCfg(PickPlaceGR1T2ObservationsCfg.PolicyCfg):
        robot_pov_cam = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("robot_pov_cam"),
                "data_type": "rgb",
                "normalize": False,
                "clone": False,
            },
        )
        left_wrist_cam = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("left_wrist_cam"),
                "data_type": "rgb",
                "normalize": False,
                "clone": False,
            },
        )
        right_wrist_cam = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("right_wrist_cam"),
                "data_type": "rgb",
                "normalize": False,
                "clone": False,
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CustomPickPlaceGR1T2MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Record policy observations, including all three RGB cameras, in episodes."""


@configclass
class CustomPickPlaceGR1T2MimicEnvCfg(CollectPickPlaceGR1T2EnvCfg, MimicEnvCfg):
    """Mimic config using the custom object and target from the GR00T collection task."""

    scene: CustomPickPlaceGR1T2MimicSceneCfg = CustomPickPlaceGR1T2MimicSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: CustomPickPlaceGR1T2MimicObservationsCfg = CustomPickPlaceGR1T2MimicObservationsCfg()
    mimic_recorder_config: CustomPickPlaceGR1T2MimicRecorderManagerCfg = (
        CustomPickPlaceGR1T2MimicRecorderManagerCfg()
    )

    # Keep Mimic resets consistent with the collection environment: randomize
    # each object's tabletop position and yaw around its configured default.
    @configclass
    class EventCfg:
        reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

        reset_object = EventTerm(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "yaw": (-1.57079, 1.57079),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        reset_object_2 = EventTerm(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.05, 0.05),
                    "y": (0.0, 0.05),
                    "yaw": (-1.57079, 1.57079),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object_2"),
            },
        )

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "custom_gr1t2_pick_place_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.num_demo_to_render = 10
        self.datagen_config.num_fail_demo_to_render = 25
        self.datagen_config.seed = 1

        # Each arm has a pick phase followed by a place phase. The task is
        # annotated manually: mark the end of the shared pick phase once while
        # viewing each arm's replay. The final place phase ends implicitly.
        for eef_name in ("left", "right"):
            self.subtask_configs[eef_name] = [
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal=f"pick_done_{eef_name}",
                    first_subtask_start_offset_range=(0, 0),
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.003,
                    num_interpolation_steps=0,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                ),
                SubTaskConfig(
                    object_ref="object_2",
                    subtask_term_signal=None,
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.003,
                    num_interpolation_steps=3,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                ),
            ]

        # The simultaneous two-hand grasp is a coordination subtask. Mimic
        # selects the same source demo, applies one shared object transform,
        # and synchronizes the subtask end for both arms.
        self.task_constraint_configs = [
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left", 0), ("right", 0)],
                constraint_type=SubTaskConstraintType.COORDINATION,
                coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
                coordination_synchronize_start=True,
            )
        ]
