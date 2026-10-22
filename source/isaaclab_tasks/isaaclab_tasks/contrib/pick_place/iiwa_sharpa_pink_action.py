# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pink IK action adapter for a single arm in the two-articulation SHARPA scene."""

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
        # This URDF contains only the seven IIWA joints. The full articulation
        # also contains 22 SHARPA joints, which are commanded directly below.
        self._isaaclab_all_joint_ids = list(range(len(self._isaaclab_controlled_joint_ids)))
        self.cfg.controller.all_joint_names = self._isaaclab_controlled_joint_names

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


@configclass
class IiwaSharpaPinkIKActionCfg(PinkInverseKinematicsActionCfg):
    """Configuration for the per-arm Pink IK action adapter."""

    class_type: type[IiwaSharpaPinkIKAction] = IiwaSharpaPinkIKAction

