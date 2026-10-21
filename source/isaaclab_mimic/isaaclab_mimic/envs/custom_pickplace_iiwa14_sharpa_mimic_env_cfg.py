# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mimic configuration for the local paired IIWA14/SHARPA pick-place task."""

from isaaclab.envs.mimic_env_cfg import (
    MimicEnvCfg,
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.contrib.pick_place.custom_pick_place_iiwa14_sharpa_env_cfg import (
    Iiwa14SharpaPickPlaceEnvCfg,
)


@configclass
class Iiwa14SharpaMimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Record joint state, observations, and Pink IK actions."""


@configclass
class Iiwa14SharpaMimicEnvCfg(Iiwa14SharpaPickPlaceEnvCfg, MimicEnvCfg):
    """Mimic task reusing the VR/Pink IK controlled IIWA14/SHARPA scene."""

    mimic_recorder_config: Iiwa14SharpaMimicRecorderManagerCfg = Iiwa14SharpaMimicRecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "custom_iiwa14_sharpa_pick_place_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.seed = 1

        for arm in ("left", "right"):
            self.subtask_configs[arm] = [
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal=f"pick_done_{arm}",
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

        self.task_constraint_configs = [
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left", 0), ("right", 0)],
                constraint_type=SubTaskConstraintType.COORDINATION,
                coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
                coordination_synchronize_start=True,
            )
        ]
