"""Convert MuJoCo MJCF object models to standalone Isaac Sim USD files.

The converter builds a temporary URDF from MJCF bodies, joints, and geoms,
then uses Isaac Lab's URDF importer. Meshes are copied next to the temporary
URDF and the resulting USD stage is flattened so each output is one file.

Run in the Isaac Lab environment:

    ./isaaclab.sh -p collect/convert_mjcf_to_usd.py --check-assets
    ./isaaclab.sh -p collect/convert_mjcf_to_usd.py --all
    ./isaaclab.sh -p collect/convert_mjcf_to_usd.py --mjcf path/to/model.xml

MJCF textures and MuJoCo-specific rendering/contact parameters are not
represented exactly by URDF. Primitive shape, mesh geometry, and the basic
joint tree are converted. A missing-resource report is available without
starting Isaac Sim via --check-assets.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[1]
DEFAULT_ASSET_ROOT = WORKSPACE / "dexmimicgen/dexmimicgen/models/assets/objects"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "assets/mjcf"

# The environment code currently refers to these MJCFs. cabinet_long.xml is
# retained as a standalone asset even though drawer_long.xml is the one loaded
# by LongDrawerObject.
DEFAULT_MODELS = (
    "cabinet.xml",
    "cabinet_long.xml",
    "drawer_long.xml",
    "coffee_base.xml",
    "coffee_body.xml",
    "coffee_lid.xml",
    "coffee_pod.xml",
    "objaverse/cup_2/model.xml",
    "objaverse/bowl_7/model.xml",
    "objaverse/mug_1/model.xml",
)

# Fixed-parameter objects created from robosuite primitives in environments.
# These definitions are independent MJCFs so each generated object becomes its
# own USD. Dimensions are MuJoCo half-sizes / radii and are expressed in meters.
def _bin_geoms(size: tuple[float, float, float], wall: float, rgba: tuple[float, ...]) -> list[dict]:
    """Approximate robosuite Bin's base and four walls with boxes."""
    x, y, z = size
    return [
        {"type": "box", "size": [x, y, wall], "pos": [0, 0, -z + wall], "rgba": rgba},
        {"type": "box", "size": [wall, y, z], "pos": [-x + wall, 0, 0], "rgba": rgba},
        {"type": "box", "size": [wall, y, z], "pos": [x - wall, 0, 0], "rgba": rgba},
        {"type": "box", "size": [x, wall, z], "pos": [0, -y + wall, 0], "rgba": rgba},
        {"type": "box", "size": [x, wall, z], "pos": [0, y - wall, 0], "rgba": rgba},
    ]


GENERATED_OBJECTS = {
    "transport_trash": {
        "freejoint": True,
        "geoms": [{"type": "box", "size": [0.02, 0.02, 0.02], "rgba": [0.7, 0.7, 0.7, 1]}],
    },
    "can_sort_can": {
        "freejoint": True,
        "geoms": [{"type": "cylinder", "size": [0.03, 0.09], "rgba": [0.58, 0.15, 0.10, 1]}],
    },
    "lift_tray_obj0": {
        "freejoint": True,
        "geoms": [{"type": "box", "size": [0.03, 0.03, 0.03], "rgba": [45 / 255, 85 / 255, 1, 1]}],
    },
    "lift_tray_obj1": {
        "freejoint": True,
        "geoms": [{"type": "box", "size": [0.03, 0.03, 0.03], "rgba": [0, 153 / 255, 92 / 255, 1]}],
    },
    "pouring_pad": {
        "geoms": [{"type": "box", "size": [0.07, 0.07, 0.01], "rgba": [0, 1, 0, 1]}],
    },
    "pouring_ball": {
        "freejoint": True,
        "geoms": [{"type": "sphere", "size": [0.025], "rgba": [0.7, 0.2, 0.2, 1]}],
    },
    "coffee_pod_holder_support": {
        "geoms": [{"type": "box", "size": [0.01, 0.0162, 0.005], "rgba": [0.839, 0.839, 0.839, 1]}],
    },
    "coffee_machine_pod_holder": {
        "freejoint": True,
        "geoms": [
            {"type": "box", "size": [0.0295, 0.0295, 0.0025], "pos": [0, 0, -0.0255], "rgba": [1, 0, 0, 1]},
            {"type": "box", "size": [0.0025, 0.0295, 0.014], "pos": [-0.027, 0, -0.0115], "rgba": [1, 0, 0, 1]},
            {"type": "box", "size": [0.0025, 0.0295, 0.014], "pos": [0.027, 0, -0.0115], "rgba": [1, 0, 0, 1]},
            {"type": "box", "size": [0.0245, 0.0025, 0.014], "pos": [0, -0.027, -0.0115], "rgba": [1, 0, 0, 1]},
            {"type": "box", "size": [0.0245, 0.0025, 0.014], "pos": [0, 0.027, -0.0115], "rgba": [1, 0, 0, 1]},
        ],
    },
    "coffee_machine_cup": {
        "geoms": [
            {"type": "cylinder", "size": [0.025, 0.025], "rgba": [0.839, 0.839, 0.839, 1]},
            {"type": "cylinder", "size": [0.025, 0.005], "pos": [0, 0, -0.015], "rgba": [0.839, 0.839, 0.839, 1]},
            {"type": "capsule", "size": [0.003, 0.0125], "pos": [0, 0.034, 0], "rgba": [0.839, 0.839, 0.839, 1]},
        ],
        "freejoint": True,
    },
    "can_sort_red_box": {"geoms": _bin_geoms((0.25, 0.2, 0.05), 0.01, (0.58, 0.15, 0.10, 1.0)), "freejoint": True},
    "can_sort_blue_box": {"geoms": _bin_geoms((0.25, 0.2, 0.05), 0.01, (0.53, 0.77, 0.95, 1.0)), "freejoint": True},
    "box_cleanup_bin": {"geoms": _bin_geoms((0.2, 0.2, 0.15), 0.01, (0.2, 0.1, 0.0, 1.0)), "freejoint": True},
    "box_cleanup_lid": {
        "geoms": [
            {"type": "box", "size": [0.075, 0.075, 0.02], "pos": [0, 0, -0.005], "rgba": [0.2, 0.1, 0.0, 1]},
            {"type": "box", "size": [0.125, 0.125, 0.03], "pos": [0, 0, 0.045], "rgba": [0.2, 0.1, 0.0, 1]},
        ],
        "freejoint": True,
    },
}


def _floats(value: str | None, default: Iterable[float]) -> list[float]:
    if value is None:
        return list(default)
    return [float(item) for item in value.replace(",", " ").split()]


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _safe_name(name: str, fallback: str) -> str:
    value = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    return value or fallback


def check_mjcf(path: Path) -> list[str]:
    """Return unresolved local file references in mesh/texture/hfield/skin."""
    path = path.resolve()
    root = ET.parse(path).getroot()
    missing: list[str] = []
    for tag in ("mesh", "texture", "hfield", "skin"):
        for element in root.findall(f".//{tag}"):
            filename = element.get("file")
            if not filename:
                continue
            candidate = (path.parent / filename).resolve()
            if not candidate.is_file():
                missing.append(f"{tag} '{filename}' -> {candidate}")
    return missing


@dataclass
class Link:
    name: str
    parent: str | None
    xyz: list[float]
    rpy: list[float]
    joint_name: str | None = None
    joint_type: str = "fixed"
    axis: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    limit: tuple[float, float] | None = None
    visuals: list[str] = field(default_factory=list)
    collisions: list[str] = field(default_factory=list)


class MjcfUrdfBuilder:
    def __init__(self, mjcf_path: Path, work_dir: Path):
        self.path = mjcf_path.resolve()
        self.work_dir = work_dir
        self.mesh_dir = work_dir / "meshes"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        self.root = ET.parse(self.path).getroot()
        self.meshes = {m.get("name"): m for m in self.root.findall("./asset/mesh")}
        self.materials = {m.get("name"): m for m in self.root.findall("./asset/material")}
        self.links: list[Link] = []
        self.counter = 0

    def _mesh_path(self, name: str) -> Path:
        mesh = self.meshes[name]
        src = (self.path.parent / mesh.attrib["file"]).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Mesh '{name}' not found: {src}")
        # URDF mesh scale supports non-uniform scaling; MuJoCo refquat is
        # composed into the visual/collision origin below when practical.
        dst = self.mesh_dir / f"{_safe_name(name, 'mesh')}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)
        scale = _floats(mesh.get("scale"), (1, 1, 1))
        return dst, scale, _floats(mesh.get("refquat"), (1, 0, 0, 0))

    def _origin(self, parent: ET.Element, element: ET.Element) -> tuple[list[float], list[float]]:
        xyz = _floats(element.get("pos"), (0, 0, 0))
        if element.get("euler"):
            rpy = _floats(element.get("euler"), (0, 0, 0))
        elif element.get("quat"):
            w, x, y, z = _floats(element.get("quat"), (1, 0, 0, 0))
            rpy = _quat_rpy(w, x, y, z)
        else:
            rpy = [0, 0, 0]
        return xyz, rpy

    def _material_rgba(self, geom: ET.Element) -> list[float]:
        rgba = geom.get("rgba")
        if rgba:
            return _floats(rgba, (0.7, 0.7, 0.7, 1))
        material = self.materials.get(geom.get("material"))
        if material is not None and material.get("rgba"):
            return _floats(material.get("rgba"), (0.7, 0.7, 0.7, 1))
        return [0.7, 0.7, 0.7, 1.0]

    def _geometry(self, geom: ET.Element, owner: Link) -> None:
        kind = geom.get("type", "sphere")
        xyz, rpy = self._origin(geom, geom)
        name = _safe_name(geom.get("name", f"geom_{self.counter}"), f"geom_{self.counter}")
        self.counter += 1
        geometry = ET.Element("geometry")
        if kind == "box":
            size = _floats(geom.get("size"), (0.01, 0.01, 0.01))
            ET.SubElement(geometry, "box", size=_fmt([2 * x for x in size]))
        elif kind == "sphere":
            ET.SubElement(geometry, "sphere", radius=str(_floats(geom.get("size"), (0.01,))[0]))
        elif kind in ("cylinder", "capsule", "ellipsoid"):
            size = _floats(geom.get("size"), (0.01, 0.01))
            radius = size[0]
            length = 2 * size[1] if len(size) > 1 else 0.02
            if kind == "capsule":
                # URDF cylinder loses hemispherical ends; visual and collision
                # keep a usable approximation.
                length += 2 * radius
            ET.SubElement(geometry, "cylinder", radius=str(radius), length=str(length))
        elif kind == "mesh":
            mesh_name = geom.get("mesh")
            if not mesh_name or mesh_name not in self.meshes:
                raise ValueError(f"Geom '{name}' refers to undefined mesh {mesh_name!r}")
            mesh_path, scale, mesh_quat = self._mesh_path(mesh_name)
            ET.SubElement(geometry, "mesh", filename=f"meshes/{mesh_path.name}", scale=_fmt(scale))
            if mesh_quat != [1, 0, 0, 0]:
                qxyz, qrpy = _origin(geom, geom)
                rpy = _quat_rpy(*mesh_quat)
                rpy = [a + b for a, b in zip(qrpy, rpy)]
        else:
            print(f"[WARN] Skipping unsupported MuJoCo geom type '{kind}' ({name})", file=sys.stderr)
            return

        visual = ET.Element("visual", name=f"{name}_visual")
        ET.SubElement(visual, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
        visual.append(ET.fromstring(ET.tostring(geometry)))
        material = ET.SubElement(visual, "material", name=f"{name}_material")
        color = self._material_rgba(geom)
        ET.SubElement(material, "color", rgba=_fmt(color))
        owner.visuals.append(ET.tostring(visual, encoding="unicode"))

        # MuJoCo often declares visual-only and collision geoms with identical
        # shapes. Keep collision-enabled geoms; if flags are absent, include it.
        if geom.get("contype", "1") == "0" and geom.get("conaffinity", "1") == "0":
            return
        collision = ET.Element("collision", name=f"{name}_collision")
        ET.SubElement(collision, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
        collision.append(ET.fromstring(ET.tostring(geometry)))
        owner.collisions.append(ET.tostring(collision, encoding="unicode"))

    def _walk_body(self, body: ET.Element, parent_name: str | None, world_pos: list[float], world_rpy: list[float]) -> None:
        name = _safe_name(body.get("name", f"body_{len(self.links)}"), f"body_{len(self.links)}")
        xyz, rpy = self._origin(body, body)
        # MJCF nested body frames are relative to their parent. URDF also stores
        # relative joint origins; world_pos/rpy are only used for worldbody roots.
        if parent_name is None:
            xyz = [a + b for a, b in zip(xyz, world_pos)]
            rpy = [a + b for a, b in zip(rpy, world_rpy)]
        joints = body.findall("joint")
        if not joints:
            link = Link(name, parent_name, xyz, rpy)
            self.links.append(link)
        else:
            current_parent = parent_name
            for joint_index, joint in enumerate(joints):
                link_name = f"{name}_joint_frame_{joint_index}"
                jtype = {"hinge": "revolute", "slide": "prismatic", "free": "floating", "ball": "floating"}.get(joint.get("type", "hinge"), "fixed")
                axis = _floats(joint.get("axis"), (0, 0, 1))
                range_value = joint.get("range")
                limit = tuple(_floats(range_value, (-math.pi, math.pi))) if range_value else None
                link = Link(link_name, current_parent, xyz if joint_index == 0 else [0, 0, 0], rpy if joint_index == 0 else [0, 0, 0], joint.get("name"), jtype, axis, limit)
                self.links.append(link)
                current_parent = link_name
        for geom in body.findall("geom"):
            self._geometry(geom, link)
        for child in body.findall("body"):
            self._walk_body(child, name, [0, 0, 0], [0, 0, 0])

    def build(self, output: Path) -> Path:
        worldbody = self.root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"No <worldbody> in {self.path}")
        world_pos = _floats(worldbody.get("pos"), (0, 0, 0))
        for body in worldbody.findall("body"):
            self._walk_body(body, None, world_pos, [0, 0, 0])
        if not self.links:
            raise ValueError(f"No body links in {self.path}")

        urdf = ET.Element("robot", name=_safe_name(self.root.get("model", self.path.stem), "mjcf_asset"))
        for link in self.links:
            node = ET.SubElement(urdf, "link", name=link.name)
            for xml in link.visuals:
                node.append(ET.fromstring(xml))
            for xml in link.collisions:
                node.append(ET.fromstring(xml))
            inertial = ET.SubElement(node, "inertial")
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value="1")
            inertia = ET.SubElement(inertial, "inertia")
            for key in ("ixx", "iyy", "izz"):
                inertia.set(key, "0.001")
            for key in ("ixy", "ixz", "iyz"):
                inertia.set(key, "0")
        for link in self.links:
            if link.parent is None:
                continue
            joint = ET.SubElement(urdf, "joint", name=link.joint_name or f"{link.name}_joint", type=link.joint_type)
            ET.SubElement(joint, "parent", link=link.parent)
            ET.SubElement(joint, "child", link=link.name)
            ET.SubElement(joint, "origin", xyz=_fmt(link.xyz), rpy=_fmt(link.rpy))
            if link.joint_type in ("revolute", "continuous", "prismatic"):
                ET.SubElement(joint, "axis", xyz=_fmt(link.axis))
                lower, upper = link.limit or ((-math.pi, math.pi) if link.joint_type == "revolute" else (-1, 1))
                ET.SubElement(joint, "limit", lower=str(lower), upper=str(upper), effort="100", velocity="10")
        ET.indent(urdf, space="  ")
        ET.ElementTree(urdf).write(output, encoding="utf-8", xml_declaration=True)
        return output


def write_generated_mjcf(name: str, spec: dict, output: Path) -> Path:
    """Write a self-contained MJCF for one programmatically defined object."""
    root = ET.Element("mujoco", model=name)
    ET.SubElement(root, "compiler", angle="radian")
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", name=name)
    if spec.get("freejoint", False):
        ET.SubElement(body, "freejoint", name=f"{name}_free")
    for index, geom_spec in enumerate(spec["geoms"]):
        attrs = {"name": f"{name}_geom_{index}", "type": geom_spec["type"]}
        if "size" in geom_spec:
            attrs["size"] = _fmt(geom_spec["size"])
        if "pos" in geom_spec:
            attrs["pos"] = _fmt(geom_spec["pos"])
        if "rgba" in geom_spec:
            attrs["rgba"] = _fmt(geom_spec["rgba"])
        ET.SubElement(body, "geom", attrs)
    ET.indent(root, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


def _quat_rpy(w: float, x: float, y: float, z: float) -> list[float]:
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return [roll, pitch, math.atan2(siny_cosp, cosy_cosp)]


def convert_one(mjcf_path: Path, output_path: Path, *, fix_base: bool, merge_fixed_joints: bool) -> None:
    # Delay Isaac Lab imports until after argument parsing and asset checks.
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from pxr import Usd

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mjcf_to_urdf_") as tmp:
        temp_dir = Path(tmp)
        urdf_path = MjcfUrdfBuilder(mjcf_path, temp_dir).build(temp_dir / "asset.urdf")
        cfg = UrdfConverterCfg(
            asset_path=str(urdf_path),
            usd_dir=str(output_path.parent),
            usd_file_name=output_path.name,
            fix_base=fix_base,
            merge_fixed_joints=merge_fixed_joints,
            force_usd_conversion=True,
            make_instanceable=False,
        )
        converter_path = Path(UrdfConverter(cfg).usd_path).resolve()
        stage = Usd.Stage.Open(str(converter_path), load=Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError(f"Could not open converted stage: {converter_path}")
        if not stage.Flatten().Export(str(output_path)):
            raise RuntimeError(f"Could not export flattened USD: {output_path}")
        if converter_path != output_path and converter_path.is_file():
            converter_path.unlink()
        for sidecar in ("payloads", "Textures"):
            sidecar_path = output_path.parent / sidecar
            if sidecar_path.is_dir():
                shutil.rmtree(sidecar_path)
    print(f"[OK] {mjcf_path} -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mjcf", type=Path, help="Convert one MJCF file")
    parser.add_argument("--all", action="store_true", help="Convert the known environment MJCF assets")
    parser.add_argument("--generated", action="store_true", help="Convert programmatically generated environment primitives")
    parser.add_argument("--check-assets", action="store_true", help="Check all known MJCFs and referenced files, without launching Isaac Sim")
    parser.add_argument("--fix-base", action="store_true", help="Import roots as fixed-base articulations")
    parser.add_argument("--merge-fixed-joints", action="store_true", help="Merge fixed joints in the URDF importer")
    # AppLauncher arguments are needed for conversion; checking files should
    # also be usable in a plain Python installation.
    args, unknown = parser.parse_known_args()
    models = [args.mjcf.resolve()] if args.mjcf else [args.asset_root / item for item in DEFAULT_MODELS]
    errors = False
    for model in models:
        if not model.is_file():
            print(f"[MISSING] MJCF: {model}")
            errors = True
            continue
        missing = check_mjcf(model)
        if missing:
            errors = True
            for resource in missing:
                print(f"[MISSING] {model}: {resource}")
        else:
            print(f"[OK] assets: {model}")
    if args.check_assets:
        return 1 if errors else 0
    if args.mjcf is None and not args.all and not args.generated:
        parser.error("Specify --mjcf PATH, --all, --generated, or --check-assets")

    # Import Isaac Lab after parse/check to allow clean diagnostics on machines
    # that do not have the simulator Python environment installed.
    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise RuntimeError("Run this conversion inside an Isaac Lab Python environment") from exc
    app_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(app_parser)
    app_args = app_parser.parse_args(unknown)
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app
    try:
        if args.all or args.mjcf:
            for model in models:
                if not model.is_file():
                    continue
                if check_mjcf(model):
                    print(f"[SKIP] unresolved resources in {model}")
                    continue
                relative = model.resolve().relative_to(args.asset_root.resolve()) if model.resolve().is_relative_to(args.asset_root.resolve()) else Path(model.stem + model.suffix)
                output = args.output_dir / relative.parent / f"{relative.stem}.usd"
                convert_one(model, output, fix_base=args.fix_base, merge_fixed_joints=args.merge_fixed_joints)
        if args.generated:
            for name, spec in GENERATED_OBJECTS.items():
                with tempfile.TemporaryDirectory(prefix="generated_mjcf_") as tmp:
                    mjcf = write_generated_mjcf(name, spec, Path(tmp) / f"{name}.xml")
                    output = args.output_dir / "generated" / f"{name}.usd"
                    convert_one(mjcf, output, fix_base=args.fix_base, merge_fixed_joints=args.merge_fixed_joints)
    finally:
        simulation_app.close()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
