# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pink IK action adapter for a single arm in the two-articulation SHARPA scene."""

import torch

from isaaclab.controllers.pink_ik import PinkIKController
from isaaclab.controllers.pink_ik.pink_tasks import FrameTask, LocalFrameTask
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.utils import configclass, math as math_utils


class IiwaSharpaPinkIKAction(PinkInverseKinematicsAction):
    """Pink IK action for one contiguous 29-value arm segment."""

    def process_actions(self, actions):
        """Read ``[position, quaternion, 22 hand joints]`` from this arm's segment."""
        self._raw_actions[:] = actions
        self._target_hand_joint_positions = actions[:, 7 : 7 + self.hand_joint_dim]

        self.base_link_frame_in_world_rf = self._get_base_link_frame_transform()
        position_env = actions[:, :3] - self._env.scene.env_origins
        target_palm_pose = math_utils.make_pose(position_env, math_utils.matrix_from_quat(actions[:, 3:7]))

        # Pink's USD-to-URDF conversion keeps the IIWA chain but omits SHARPA
        # frames. Derive the fixed palm-to-link_7 transform from the USD-backed
        # articulation, then convert the commanded palm pose to link_7.
        articulation = self._env.scene[self.cfg.controller.articulation_name]
        body_names = articulation.data.body_names
        ik_frame_index = body_names.index("iiwa14_link_7")
        palm_frame_name = next((name for name in body_names if name.endswith("hand_C_MC")), None)
        if palm_frame_name is None:
            raise ValueError(f"No SHARPA palm frame ending in 'hand_C_MC' in articulation bodies: {body_names}")
        palm_frame_index = body_names.index(palm_frame_name)
        body_pose_w = articulation.data.body_link_pose_w.torch
        ik_frame_pose_w = math_utils.make_pose(
            body_pose_w[:, ik_frame_index, :3], math_utils.matrix_from_quat(body_pose_w[:, ik_frame_index, 3:7])
        )
        palm_frame_pose_w = math_utils.make_pose(
            body_pose_w[:, palm_frame_index, :3], math_utils.matrix_from_quat(body_pose_w[:, palm_frame_index, 3:7])
        )
        ik_frame_inv_w = math_utils.pose_inv(ik_frame_pose_w)
        ik_frame_to_palm = math_utils.pose_in_A_to_pose_in_B(palm_frame_pose_w, ik_frame_inv_w)
        palm_to_ik_frame = math_utils.pose_inv(ik_frame_to_palm)
        target_ik_frame_pose = torch.matmul(target_palm_pose, palm_to_ik_frame)

        base_link_inv = math_utils.pose_inv(self.base_link_frame_in_world_rf)
        target_in_base = math_utils.pose_in_A_to_pose_in_B(target_ik_frame_pose, base_link_inv)
        position, rotation = math_utils.unmake_pose(target_in_base)

        for env_index, controller in enumerate(self._ik_controllers):
            for task in controller.cfg.variable_input_tasks:
                if isinstance(task, LocalFrameTask):
                    target = task.transform_target_to_base
                elif isinstance(task, FrameTask):
                    target = task.transform_target_to_world
                else:
                    continue
                target.translation = position[env_index].cpu().numpy()
                target.rotation = rotation[env_index].cpu().numpy()
                task.set_target(target)

    def _initialize_ik_controllers(self) -> None:
        assert self._env.num_envs > 0, "Number of environments specified are less than 1."
        robot_cfg = getattr(self._env.scene.cfg, self.cfg.controller.articulation_name)
        self._ik_controllers = [
            PinkIKController(
                cfg=self.cfg.controller.copy(),
                robot_cfg=robot_cfg,
                device=self.device,
                controlled_joint_indices=self._isaaclab_controlled_joint_ids,
            )
            for _ in range(self._env.num_envs)
        ]


@configclass
class IiwaSharpaPinkIKActionCfg(PinkInverseKinematicsActionCfg):
    """Configuration for the per-arm Pink IK action adapter."""

    class_type: type[IiwaSharpaPinkIKAction] = IiwaSharpaPinkIKAction

