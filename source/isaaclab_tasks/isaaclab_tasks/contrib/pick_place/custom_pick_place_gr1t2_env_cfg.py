# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR1T2 pick-place task with the two local ACG assets on the table."""

from pathlib import Path

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.contrib.pick_place import mdp

from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import (
    PickPlaceGR1T2EnvCfg,
    PickPlaceGR1T2SceneCfg,
)


_ASSET_DIR = Path(__file__).resolve().parent / "assets"
_OBJECT_1_USD = _ASSET_DIR / "box_cleanup_lid.usd"
_OBJECT_2_USD = _ASSET_DIR / "box_cleanup_bin.usd"

# Adjust these world-space coordinates [m] to move the assets on the tabletop.
# X/Y set the tabletop location and Z sets the root prim height. Restart the
# environment after editing; the reset event returns each object to this pose.
OBJECT_1_POSITION = (-0.30, 0.35, 1.01)
OBJECT_2_POSITION = (0.1, 0.35, 1.11)

# Uniform USD scale factors. Change these values to adjust each object's size.
OBJECT_1_SCALE = (0.8, 0.8, 0.8)
OBJECT_2_SCALE = (0.5, 0.5, 0.5)
_BASE_SCENE_CFG = PickPlaceGR1T2SceneCfg()


def _spawn_packing_table_without_container(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the standard table and remove its default container prop."""
    table_prim = sim_utils.spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    stage = sim_utils.get_current_stage()
    # spawn_from_usd clones the table before returning, so remove the prop from
    # every cloned environment, rather than only from the source table prim.
    table_paths = sim_utils.find_matching_prim_paths(prim_path)
    found_container = False
    for table_path in table_paths:
        table = stage.GetPrimAtPath(table_path)
        for child in Usd.PrimRange(table):
            if child.GetName().casefold() == "container_h20":
                # Deactivate instead of removing: this suppresses referenced USD content.
                child.SetActive(False)
                found_container = True
    if not found_container:
        child_paths = [
            child.GetPath().pathString
            for table_path in table_paths
            for child in Usd.PrimRange(stage.GetPrimAtPath(table_path))
        ]
        raise RuntimeError(f"Could not find container_h20 below packing table(s). Descendants: {child_paths}")
    return table_prim

@configclass
class CollectSceneCfg(PickPlaceGR1T2SceneCfg):
    """GR1T2 pick-place scene with two configurable custom objects."""

    # The inherited packing table USD is only a table; it contains no basket.
    # The inherited steering wheel is replaced by the custom object below.

    packing_table = _BASE_SCENE_CFG.packing_table.replace(
        spawn=_BASE_SCENE_CFG.packing_table.spawn.replace(
            func=_spawn_packing_table_without_container,
            make_uninstanceable=True,
        )
    )

    # Operated object; this retains the stock observation, reset, and success API.
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_1_POSITION),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_OBJECT_1_USD),
            scale=OBJECT_1_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )
    object_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_2_POSITION),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_OBJECT_2_USD),
            scale=OBJECT_2_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # Stock sensors filter against a nested steering wheel prim. The imported
    # assets author their rigid body at their root, so target the fixture root.
    left_hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*L_(index|middle|ring|pinky|thumb)[^/]*_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        history_length=3,
    )
    right_hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*R_(index|middle|ring|pinky|thumb)[^/]*_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        history_length=3,
    )


@configclass
class CollectPickPlaceGR1T2EnvCfg(PickPlaceGR1T2EnvCfg):
    """Stock GR1T2 XR pick-place workflow using the local fixture and peg."""

    scene: CollectSceneCfg = CollectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    def __post_init__(self):
        super().__post_init__()
        # Contact sensors in the inherited scene now measure the fixture at Object.
        self.haptic_feedback.left_sensor_name = "left_hand_contact"
        self.haptic_feedback.right_sensor_name = "right_hand_contact"
        # Success means object 1 has been moved close to object 2, has settled,
        # and the right wrist has retracted, following the stock task predicate.
        self.terminations.success = DoneTerm(
            func=mdp.task_done_object_near_target,
            params={
                "task_link_name": "right_hand_roll_link",
                "object_cfg": SceneEntityCfg("object"),
                "target_cfg": SceneEntityCfg("object_2"),
            },
        )
