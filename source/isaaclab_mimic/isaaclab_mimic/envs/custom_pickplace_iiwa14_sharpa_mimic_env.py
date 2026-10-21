# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mimic action/pose adapter for the paired IIWA14/SHARPA environment."""

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class Iiwa14SharpaMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic interface matching the per-arm pose-plus-finger Pink action layout."""

    _HAND_DOF = 22
    _POSE_DIM = 7
    _ARM_ACTION_DIM = _POSE_DIM + _HAND_DOF
    _ACTION_DIM = 2 * _ARM_ACTION_DIM
    _ARMS = ("left", "right")

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if eef_name not in ("left", "right"):
            raise ValueError(f"Unknown end effector {eef_name!r}; expected 'left' or 'right'.")
        if env_ids is None:
            env_ids = slice(None)
        elif isinstance(env_ids, int):
            # Keep the environment dimension so the result always has shape (N, 4, 4).
            env_ids = [env_ids]

        asset = self.scene[f"{eef_name}_arm"]
        body_name = f"{eef_name}_hand_C_MC"
        try:
            body_idx = asset.data.body_names.index(body_name)
        except ValueError as exc:
            raise ValueError(
                f"Could not find end-effector body {body_name!r} on scene asset {eef_name + '_arm'!r}. "
                f"Available bodies: {asset.data.body_names}"
            ) from exc
        link_pose_w = asset.data.body_link_pose_w.torch[env_ids, body_idx]
        env_origins = self.scene.env_origins[env_ids]
        pose_env = PoseUtils.make_pose(
            link_pose_w[..., :3] - env_origins,
            PoseUtils.matrix_from_quat(link_pose_w[..., 3:7]),
        )
        return pose_env

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        del env_id
        action_parts = []
        for arm in self._ARMS:
            if arm not in target_eef_pose_dict:
                raise KeyError(f"Missing {arm!r} end-effector pose in target_eef_pose_dict.")
            if arm not in gripper_action_dict:
                raise KeyError(f"Missing {arm!r} hand target in gripper_action_dict.")
            target_pose = target_eef_pose_dict[arm]
            position, rotation = PoseUtils.unmake_pose(target_pose)
            quat = PoseUtils.quat_unique(PoseUtils.quat_from_matrix(rotation))
            if action_noise_dict is not None:
                if arm not in action_noise_dict:
                    raise KeyError(f"Missing action noise scale for {arm!r}.")
                noise_scale = action_noise_dict[arm]
                position = position + noise_scale * torch.randn_like(position)
                quat = quat + noise_scale * torch.randn_like(quat)

            hand_targets = gripper_action_dict[arm].reshape(-1)
            if hand_targets.numel() != self._HAND_DOF:
                raise ValueError(f"Expected {self._HAND_DOF} {arm} SHARPA finger targets, got {hand_targets.numel()}.")
            action_parts.extend((position.reshape(-1), quat.reshape(-1), hand_targets))
        return torch.cat(action_parts, dim=-1).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        if action.ndim == 0 or action.shape[-1] != self._ACTION_DIM:
            width = action.shape[-1] if action.ndim else 0
            raise ValueError(f"Expected action width {self._ACTION_DIM}, got {width}.")
        poses = {}
        for arm, start in (("left", 0), ("right", self._ARM_ACTION_DIM)):
            position = action[..., start : start + 3]
            quat = action[..., start + 3 : start + 7]
            poses[arm] = PoseUtils.make_pose(position, PoseUtils.matrix_from_quat(quat))
        return poses

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if actions.ndim == 0 or actions.shape[-1] != self._ACTION_DIM:
            width = actions.shape[-1] if actions.ndim else 0
            raise ValueError(f"Expected action width {self._ACTION_DIM}, got {width}.")
        right_start = self._ARM_ACTION_DIM
        return {
            "left": actions[..., self._POSE_DIM : self._ARM_ACTION_DIM],
            "right": actions[..., right_start + self._POSE_DIM : right_start + self._ARM_ACTION_DIM],
        }
