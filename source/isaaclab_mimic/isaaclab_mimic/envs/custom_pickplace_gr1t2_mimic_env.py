# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mimic runtime wrapper for the local GR1T2 pick and place task."""

from isaaclab_mimic.envs.pickplace_gr1t2_mimic_env import PickPlaceGR1T2MimicEnv


class CustomPickPlaceGR1T2MimicEnv(PickPlaceGR1T2MimicEnv):
    """GR1T2 Mimic API with the custom collection scene and success condition."""
