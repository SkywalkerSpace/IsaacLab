# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-place task scene using the local paired IIWA14/SHARPA USD assets."""

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import yaml

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import (
    DampingTaskCfg,
    FrameTaskCfg,
    NullSpacePostureTaskCfg,
    PinkIKControllerCfg,
)
from isaaclab.controllers.utils import convert_usd_to_urdf
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg, EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
from isaaclab_teleop.xr_cfg import XrCfg

from . import mdp
from .iiwa_sharpa_pink_action import IiwaSharpaPinkIKActionCfg
from .custom_pick_place_gr1t2_env_cfg import (
    OBJECT_1_POSITION,
    OBJECT_1_SCALE,
    OBJECT_2_POSITION,
    OBJECT_2_SCALE,
    _OBJECT_1_USD,
    _OBJECT_2_USD,
)

# Marker used by environment utilities to recognize this as a teleoperation task.
_TELEOP_AVAILABLE = True


_ISAACLAB_ROOT = Path(__file__).resolve().parents[4]
_LEFT_ARM_USD = _ISAACLAB_ROOT / "convert/assets/iiwa14_left_sharpa_adjusted_restricted.usd"
_RIGHT_ARM_USD = _ISAACLAB_ROOT / "convert/assets/iiwa14_right_sharpa_mirror_restricted.usd"
_IK_URDF_OUTPUT_DIR = Path(tempfile.gettempdir()) / "isaaclab_sharpa_ik_urdf"
_DEX_CONFIG_DIR = Path(__file__).resolve().parent / "assets/dex_retargeting"

LEFT_HAND_JOINTS = [
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE", "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA", "left_pinky_PIP", "left_pinky_DIP",
]
RIGHT_HAND_JOINTS = [
    "right_1_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE", "right_thumb_MCP_AA", "right_thumb_IP",
    "right_2_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_index_DIP",
    "right_3_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP", "right_middle_DIP",
    "right_4_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP", "right_ring_DIP",
    "right_5_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA", "right_pinky_PIP", "right_pinky_DIP",
]


def build_iiwa14_sharpa_teleop_pipeline():
    """Build a dual wrist-pose and SHARPA finger-retargeting pipeline for OpenXR hands."""
    from isaacteleop.retargeters import DexHandRetargeter, DexHandRetargeterConfig, Se3AbsRetargeter, Se3RetargeterConfig
    from isaacteleop.retargeters import TensorReorderer
    from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    retargeting_dir = Path(tempfile.gettempdir()) / "isaaclab_sharpa_retargeting"
    retargeting_dir.mkdir(parents=True, exist_ok=True)

    def usd_transform_to_origin(matrix):
        """Convert a USD row-vector transform into a URDF xyz/rpy origin."""
        from pxr import Gf

        transform = Gf.Transform(matrix)
        xyz = transform.GetTranslation()
        euler = transform.GetRotation().GetEuler()
        return " ".join(f"{float(value):.9g}" for value in xyz), " ".join(
            f"{float(value) * 3.141592653589793 / 180.0:.9g}" for value in euler
        )

    def add_origin(parent, xyz, rpy):
        ET.SubElement(parent, "origin", xyz=xyz, rpy=rpy)

    def augment_urdf_with_usd_hand(arm_usd: Path, urdf_path: str, side: str, hand_joint_names: list[str]):
        """Add the SHARPA kinematic chain to the arm URDF, using USD physics joints as source."""
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.Open(str(arm_usd))
        if stage is None:
            raise RuntimeError(f"Could not open SHARPA USD asset: {arm_usd}")
        root = ET.parse(urdf_path)
        robot = root.getroot()
        existing_links = {link.get("name") for link in robot.findall("link")}
        existing_joints = {joint.get("name") for joint in robot.findall("joint")}
        link7 = "iiwa14_link_7"
        if link7 not in existing_links:
            raise RuntimeError(f"Arm URDF {urdf_path} does not contain expected parent link {link7}")

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        body_names = {}
        joint_prims = {}
        for prim in stage.Traverse():
            if prim.GetTypeName().startswith("Physics") and prim.GetTypeName().endswith("Joint"):
                joint_prims[prim.GetName()] = prim
        expected = set(hand_joint_names)
        missing_usd_joints = sorted(expected - set(joint_prims))
        if missing_usd_joints:
            raise RuntimeError(f"USD {arm_usd.name} does not contain configured hand joints: {missing_usd_joints}")

        hand_root_name = f"{side}_hand_C_MC"
        hand_root_prim = next(
            (prim for prim in stage.Traverse() if prim.GetName() == hand_root_name), None
        )
        arm_link_prim = next(
            (prim for prim in stage.Traverse() if prim.GetName() == link7), None
        )
        if hand_root_prim is None or arm_link_prim is None:
            raise RuntimeError(f"USD {arm_usd.name} is missing the arm or palm frame")
        bridge = cache.GetLocalToWorldTransform(hand_root_prim) * cache.GetLocalToWorldTransform(
            arm_link_prim
        ).GetInverse()
        bridge_xyz, bridge_rpy = usd_transform_to_origin(bridge)
        if hand_root_name not in existing_links:
            ET.SubElement(robot, "link", name=hand_root_name)
            existing_links.add(hand_root_name)
        if f"{side}_arm_to_palm" not in existing_joints:
            bridge_joint = ET.SubElement(robot, "joint", name=f"{side}_arm_to_palm", type="fixed")
            ET.SubElement(bridge_joint, "parent", link=link7)
            ET.SubElement(bridge_joint, "child", link=hand_root_name)
            add_origin(bridge_joint, bridge_xyz, bridge_rpy)

        for name in hand_joint_names:
            if name in existing_joints:
                continue
            prim = joint_prims[name]
            body0 = prim.GetRelationship("physics:body0").GetTargets()[0]
            body1 = prim.GetRelationship("physics:body1").GetTargets()[0]
            parent_name = stage.GetPrimAtPath(body0).GetName()
            child_name = stage.GetPrimAtPath(body1).GetName()
            body_names[parent_name] = body0
            body_names[child_name] = body1
            for body_name in (parent_name, child_name):
                if body_name not in existing_links:
                    ET.SubElement(robot, "link", name=body_name)
                    existing_links.add(body_name)

            pos0 = prim.GetAttribute("physics:localPos0").Get()
            rot0 = prim.GetAttribute("physics:localRot0").Get()
            pos1 = prim.GetAttribute("physics:localPos1").Get()
            rot1 = prim.GetAttribute("physics:localRot1").Get()
            quat0 = Gf.Quatd(float(rot0.GetReal()), Gf.Vec3d(*map(float, rot0.GetImaginary())))
            quat1 = Gf.Quatd(float(rot1.GetReal()), Gf.Vec3d(*map(float, rot1.GetImaginary())))
            parent_transform = Gf.Transform()
            parent_transform.SetTranslation(Gf.Vec3d(*map(float, pos0)))
            parent_transform.SetRotation(Gf.Rotation(quat0))
            origin_xyz, origin_rpy = usd_transform_to_origin(parent_transform.GetMatrix())

            frame_name = f"dex_joint_frame_{name}"
            ET.SubElement(robot, "link", name=frame_name)
            revolute = ET.SubElement(robot, "joint", name=name, type="revolute")
            ET.SubElement(revolute, "parent", link=parent_name)
            ET.SubElement(revolute, "child", link=frame_name)
            add_origin(revolute, origin_xyz, origin_rpy)
            axis = str(prim.GetAttribute("physics:axis").Get()).upper()
            ET.SubElement(revolute, "axis", xyz={"X": "1 0 0", "Y": "0 1 0", "Z": "0 0 1"}[axis])
            lower = float(prim.GetAttribute("physics:lowerLimit").Get()) * 3.141592653589793 / 180.0
            upper = float(prim.GetAttribute("physics:upperLimit").Get()) * 3.141592653589793 / 180.0
            limit = ET.SubElement(revolute, "limit", lower=f"{lower:.9g}", upper=f"{upper:.9g}")
            limit.set("effort", "3.3")
            limit.set("velocity", "12.0")
            existing_joints.add(name)

            child_transform = Gf.Transform()
            child_transform.SetTranslation(Gf.Vec3d(*map(float, pos1)))
            child_transform.SetRotation(Gf.Rotation(quat1))
            fixed_xyz, fixed_rpy = usd_transform_to_origin(child_transform.GetMatrix().GetInverse())
            fixed = ET.SubElement(robot, "joint", name=f"{name}_body_offset", type="fixed")
            ET.SubElement(fixed, "parent", link=frame_name)
            ET.SubElement(fixed, "child", link=child_name)
            add_origin(fixed, fixed_xyz, fixed_rpy)

        # DexPilot expects fingertip frame names. The USD hand calls the same
        # terminal rigid bodies *_DP, so add fixed aliases at those USD frames.
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            terminal_name = f"{side}_{finger}_DP"
            terminal_prim = next((prim for prim in stage.Traverse() if prim.GetName() == terminal_name), None)
            alias_name = f"{side}_{finger}_fingertip"
            if terminal_prim is None:
                raise RuntimeError(f"USD {arm_usd.name} does not contain fingertip body {terminal_name}")
            if terminal_name not in existing_links:
                ET.SubElement(robot, "link", name=terminal_name)
                existing_links.add(terminal_name)
            ET.SubElement(robot, "link", name=alias_name)
            alias_joint = ET.SubElement(robot, "joint", name=f"{alias_name}_fixed", type="fixed")
            ET.SubElement(alias_joint, "parent", link=terminal_name)
            ET.SubElement(alias_joint, "child", link=alias_name)
            add_origin(alias_joint, "0 0 0", "0 0 0")

        root.write(urdf_path, encoding="utf-8", xml_declaration=True)

    def prepare_dex_config(arm_usd: Path, config_name: str) -> tuple[Path, Path]:
        urdf_path, _ = convert_usd_to_urdf(
            str(arm_usd), str(retargeting_dir / arm_usd.stem), force_conversion=True
        )
        from isaaclab.controllers.utils import _sanitize_urdf_for_pinocchio

        _sanitize_urdf_for_pinocchio(urdf_path)
        side = "left" if "left" in config_name else "right"
        hand_joints = LEFT_HAND_JOINTS if side == "left" else RIGHT_HAND_JOINTS
        augment_urdf_with_usd_hand(arm_usd, urdf_path, side, hand_joints)
        # Dex-retargeting requires a scalar-coordinate Pinocchio model
        # (nq == nv). The USD exporter may represent the robot root with a
        # floating/planar/spherical joint. Fix those special joints in this
        # temporary retargeting model; hand revolute joints and USD-derived
        # geometry remain unchanged.
        tree = ET.parse(urdf_path)
        changed = False
        for joint in tree.getroot().findall("joint"):
            if joint.get("type") in {"floating", "planar", "spherical"}:
                joint.set("type", "fixed")
                changed = True
        if changed:
            tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

        exported_joint_names = {joint.get("name") for joint in tree.getroot().findall("joint")}
        configured_joint_names = LEFT_HAND_JOINTS if "left" in config_name else RIGHT_HAND_JOINTS
        missing_joint_names = sorted(set(configured_joint_names) - exported_joint_names)
        if missing_joint_names:
            raise RuntimeError(
                f"USD-to-URDF export for {arm_usd.name} is missing SHARPA joints: {missing_joint_names}. "
                "Check the USD robotJoints/robotLinks relationships."
            )

        config = yaml.safe_load((_DEX_CONFIG_DIR / config_name).read_text())
        config["retargeting"]["urdf_path"] = urdf_path
        config_path = retargeting_dir / config_name
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        return Path(urdf_path), config_path

    left_urdf, left_dex_config = prepare_dex_config(_LEFT_ARM_USD, "sharpa_left_dexpilot.yml")
    right_urdf, right_dex_config = prepare_dex_config(_RIGHT_ARM_USD, "sharpa_right_dexpilot.yml")

    hands = HandsSource(name="hands")
    world_T_anchor = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(world_T_anchor.output(ValueInput.VALUE))

    left_pose = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=HandsSource.LEFT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            # Target the SHARPA palm frame: the USD's +15 deg mount and -90 deg
            # wrist fixed rotations combine to -75 deg from the IIWA tool frame.
            target_offset_yaw=-75.0,
        ),
        name="left_ee_pose",
    ).connect({HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT)})
    right_pose = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=HandsSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            # The right USD mounts the palm at +15 deg from the IIWA tool frame.
            target_offset_yaw=15.0,
        ),
        name="right_ee_pose",
    ).connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})

    operator2mano = (0, -1, 0, -1, 0, 0, 0, 0, -1)
    left_hand = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=str(left_dex_config),
            hand_urdf=str(left_urdf),
            hand_joint_names=LEFT_HAND_JOINTS,
            hand_side="left",
            handtracking_to_baselink_frame_transform=operator2mano,
        ),
        name="left_sharpa_hand",
    ).connect({HandsSource.LEFT: hands.output(HandsSource.LEFT)})
    right_hand = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=str(right_dex_config),
            hand_urdf=str(right_urdf),
            hand_joint_names=RIGHT_HAND_JOINTS,
            hand_side="right",
            handtracking_to_baselink_frame_transform=operator2mano,
        ),
        name="right_sharpa_hand",
    ).connect({HandsSource.RIGHT: hands.output(HandsSource.RIGHT)})

    left_pose_keys = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_pose_keys = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]
    output_order = left_pose_keys + LEFT_HAND_JOINTS + right_pose_keys + RIGHT_HAND_JOINTS
    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_pose_keys,
            "left_hand": LEFT_HAND_JOINTS,
            "right_ee_pose": right_pose_keys,
            "right_hand": RIGHT_HAND_JOINTS,
        },
        output_order=output_order,
        name="iiwa_sharpa_action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "left_hand": "scalar",
            "right_ee_pose": "array",
            "right_hand": "scalar",
        },
    ).connect(
        {
            "left_ee_pose": left_pose.output("ee_pose"),
            "left_hand": left_hand.output("hand_joints"),
            "right_ee_pose": right_pose.output("ee_pose"),
            "right_hand": right_hand.output("hand_joints"),
        }
    )
    return OutputCombiner({"action": reorderer.output("output")}), [left_hand, right_hand]


def _make_arm_ik_action(arm_name: str, eef_name: str, hand_joint_names: list[str]):
    """Create the one-frame Pink action config for one arm articulation."""
    return IiwaSharpaPinkIKActionCfg(
        asset_name=arm_name,
        pink_controlled_joint_names=[f"iiwa14_joint_{index}" for index in range(1, 8)],
        hand_joint_names=hand_joint_names,
        target_eef_link_names={f"{arm_name}_wrist": eef_name},
        enable_gravity_compensation=False,
        controller=PinkIKControllerCfg(
            articulation_name=arm_name,
            base_link_name="iiwa14_link_0",
            urdf_output_dir=str(_IK_URDF_OUTPUT_DIR / arm_name),
            num_hand_joints=len(hand_joint_names),
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame=eef_name,
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=10,
                    gain=0.5,
                ),
                DampingTaskCfg(cost=0.5),
                NullSpacePostureTaskCfg(
                    cost=0.5,
                    lm_damping=1.0,
                    controlled_frames=[eef_name],
                    controlled_joints=[f"iiwa14_joint_{index}" for index in range(1, 8)],
                ),
            ],
            fixed_input_tasks=[],
        ),
    )


@configclass
class Iiwa14SharpaPickPlaceSceneCfg(InteractiveSceneCfg):
    """Two independently loaded arm/hand articulations on a shared pedestal."""

    # Pedestal top is at z=0.86 m. Arm origins are separated across its width.
    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.43)),
        spawn=sim_utils.CuboidCfg(
            size=(0.65, 0.55, 0.86),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.24, 0.28)),
        ),
    )
    left_arm: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LeftArm",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_LEFT_ARM_USD), activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(-0.22, 0.0, 0.86), joint_pos={".*": 0.0}),
        actuators={"arm_and_hand": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=400.0, damping=40.0)},
    )
    right_arm: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RightArm",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_RIGHT_ARM_USD), activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.22, 0.0, 0.86), joint_pos={".*": 0.0}),
        actuators={"arm_and_hand": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=400.0, damping=40.0)},
    )
    packing_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.55, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        ),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_1_POSITION),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_OBJECT_1_USD), scale=OBJECT_1_SCALE, rigid_props=sim_utils.UsdPhysicsRigidBodyCfg()
        ),
    )
    object_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_2_POSITION),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_OBJECT_2_USD), scale=OBJECT_2_SCALE, rigid_props=sim_utils.UsdPhysicsRigidBodyCfg()
        ),
    )
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    )


@configclass
class Iiwa14SharpaActionsCfg:
    """One wrist-pose Pink IK action for each arm and its SHARPA hand."""

    # The USD contains the palm, but USD-to-URDF export currently omits it from
    # Pink's frame list. Control the matching last IIWA link and convert palm
    # targets to that link in IiwaSharpaPinkIKAction.
    left_arm: ActionTermCfg = _make_arm_ik_action(
        "left_arm", "iiwa14_link_7", LEFT_HAND_JOINTS
    )
    right_arm: ActionTermCfg = _make_arm_ik_action(
        "right_arm", "iiwa14_link_7", RIGHT_HAND_JOINTS
    )


@configclass
class Iiwa14SharpaObservationsCfg:
    """Joint state and object state observations, without GR1T2 link assumptions."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=base_mdp.last_action)
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        left_arm_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("left_arm")})
        right_arm_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("right_arm")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class Iiwa14SharpaTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class Iiwa14SharpaEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-1.57079, 1.57079)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_object_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (0.0, 0.05), "yaw": (-1.57079, 1.57079)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object_2"),
        },
    )


@configclass
class Iiwa14SharpaPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """State-based pick-place environment for paired IIWA14/SHARPA assets."""

    scene: Iiwa14SharpaPickPlaceSceneCfg = Iiwa14SharpaPickPlaceSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    actions: Iiwa14SharpaActionsCfg = Iiwa14SharpaActionsCfg()
    observations: Iiwa14SharpaObservationsCfg = Iiwa14SharpaObservationsCfg()
    terminations: Iiwa14SharpaTerminationsCfg = Iiwa14SharpaTerminationsCfg()
    events: Iiwa14SharpaEventsCfg = Iiwa14SharpaEventsCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 6
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # Keep Pink's kinematics model tied to the exact USDs loaded in-scene.
        self.actions.left_arm.controller.usd_path = self.scene.left_arm.spawn.usd_path
        self.actions.right_arm.controller.usd_path = self.scene.right_arm.spawn.usd_path
        self.xr = XrCfg(anchor_pos=(0.0, 0.0, 0.0), anchor_rot=(0.0, 0.0, 0.0, 1.0))
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: build_iiwa14_sharpa_teleop_pipeline()[0],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )
