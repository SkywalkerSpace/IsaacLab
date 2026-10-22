# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pink IK action adapter for a single arm in the two-articulation SHARPA scene."""

import numpy as np

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

        base_link_inv = math_utils.pose_inv(self.base_link_frame_in_world_rf)
        target_in_base = math_utils.pose_in_A_to_pose_in_B(target_palm_pose, base_link_inv)
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

    def _initialize_joint_info(self) -> None:
        """Keep Pink's arm joint space separate from the articulation's hand joints."""
        self._isaaclab_controlled_joint_ids, self._isaaclab_controlled_joint_names = self._asset.find_joints(
            self.cfg.pink_controlled_joint_names
        )
        self.cfg.controller.joint_names = self._isaaclab_controlled_joint_names
        # Pink's URDF has the seven arm joints; Isaac's articulation also has
        # 22 hand joints. Pink's mapping indexes the full state passed by the
        # parent action term, so retain the full articulation name list here.
        self._isaaclab_all_joint_ids = list(range(len(self._asset.data.joint_names)))
        self.cfg.controller.all_joint_names = self._asset.data.joint_names

        self._hand_joint_ids, self._hand_joint_names = self._asset.find_joints(self.cfg.hand_joint_names)
        self._controlled_joint_ids = self._isaaclab_controlled_joint_ids + self._hand_joint_ids
        self._controlled_joint_names = self._isaaclab_controlled_joint_names + self._hand_joint_names

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
        # Pink's generic mapper assumes its model joints all exist in Isaac.
        # The paired URDF omits hand joints, so construct only the valid map
        # entries while keeping articulation indexing for the full state.
        for controller in self._ik_controllers:
            pink_names = controller.pink_configuration.all_joint_names_pinocchio_order
            isaac_names = self.cfg.controller.all_joint_names
            pink_controlled_names = controller.pink_configuration.controlled_joint_names_pinocchio_order
            isaac_controlled_names = self.cfg.controller.joint_names
            controller.isaac_lab_to_pink_ordering = np.array(
                [isaac_names.index(name) if name in isaac_names else -1 for name in pink_names]
            )
            controller.pink_to_isaac_lab_ordering = np.array(
                [pink_names.index(name) if name in pink_names else -1 for name in isaac_names]
            )
            controller.isaac_lab_to_pink_controlled_ordering = np.array(
                [isaac_controlled_names.index(name) for name in pink_controlled_names]
            )
            controller.pink_to_isaac_lab_controlled_ordering = np.array(
                [pink_controlled_names.index(name) for name in isaac_controlled_names]
            )


@configclass
class IiwaSharpaPinkIKActionCfg(PinkInverseKinematicsActionCfg):
    """Configuration for the per-arm Pink IK action adapter."""

    class_type: type[IiwaSharpaPinkIKAction] = IiwaSharpaPinkIKAction

