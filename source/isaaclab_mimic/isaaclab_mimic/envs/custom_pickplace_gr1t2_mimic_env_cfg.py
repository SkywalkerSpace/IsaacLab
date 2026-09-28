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
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.pick_place.custom_pick_place_gr1t2_env_cfg import (
    CollectPickPlaceGR1T2EnvCfg,
)


@configclass
class CustomPickPlaceGR1T2MimicEnvCfg(CollectPickPlaceGR1T2EnvCfg, MimicEnvCfg):
    """Mimic config using the custom object and target from the GR00T collection task."""

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
