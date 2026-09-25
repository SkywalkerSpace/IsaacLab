"""Convert the frozen-eval peg and graspable fixture URDFs to standalone USD files.

Run from an Isaac Lab environment, for example::

    ./isaaclab.sh -p /path/to/ACG/libs/IsaacLab/convert_urdfs.py

Each output is a single flattened binary USD file with payloads and references
resolved into the file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
ASSET_ROOT = (
    WORKSPACE_ROOT
    / "peg_screwing_dual_migration_20260918"
    / "play2perfect_left_official_right_mirror"
)
DEFAULT_PEG_URDF = ASSET_ROOT / "bimanual_tpeg_frozen_eval" / "assets" / "lpeg_matchedmass.urdf"
DEFAULT_FIXTURE_URDF = (
    ASSET_ROOT / "assets" / "fixture_graspable" / "hole_tol0p25mm_right_training_graspable.urdf"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "assets"


DEFAULT_PEG_URDF = "./lpeg_matchedmass.urdf"
DEFAULT_FIXTURE_URDF = (
    "./hole_tol0p25mm_right_training_graspable.urdf"
)
DEFAULT_OUTPUT_DIR = "./"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--peg-urdf", type=Path, default=DEFAULT_PEG_URDF)
parser.add_argument("--fixture-urdf", type=Path, default=DEFAULT_FIXTURE_URDF)
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
parser.add_argument(
    "--fix-base",
    action="store_true",
    help="Import both assets with fixed bases (default: leave the base unfixed).",
)
parser.add_argument(
    "--merge-fixed-joints",
    action="store_true",
    help="Merge fixed joints during import (default: preserve all links and frames).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from pxr import Usd


def convert_urdf(urdf_path: Path, output_path: Path) -> Path:
    """Convert one URDF to a standalone, flattened USD file."""
    urdf_path = urdf_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF does not exist: {urdf_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(urdf_path),
            usd_dir=str(output_path.parent),
            usd_file_name=output_path.name,
            fix_base=args_cli.fix_base,
            merge_fixed_joints=args_cli.merge_fixed_joints,
            force_usd_conversion=True,
            make_instanceable=False,
        )
    )
    converter_path = Path(converter.usd_path).resolve()

    # Load all payloads before flattening so geometry and physics are embedded.
    stage = Usd.Stage.Open(str(converter_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open converted USD: {converter_path}")
    flattened_layer = stage.Flatten()
    if not flattened_layer.Export(str(output_path)):
        raise RuntimeError(f"Could not write flattened USD: {output_path}")

    # Remove converter sidecar files: the flattened output no longer references them.
    import shutil

    for sidecar_name in ("payloads", "Textures"):
        sidecar_dir = output_path.parent / sidecar_name
        if sidecar_dir.is_dir():
            shutil.rmtree(sidecar_dir)
    if converter_path != output_path and converter_path.is_file():
        converter_path.unlink()

    print(f"[INFO] {urdf_path} -> {output_path}")
    return output_path


def main() -> None:
    output_dir = args_cli.output_dir.expanduser().resolve()
    convert_urdf(
        args_cli.peg_urdf,
        output_dir / "lpeg_matchedmass" / "lpeg_matchedmass.usd",
    )
    convert_urdf(
        args_cli.fixture_urdf,
        output_dir
        / "hole_tol0p25mm_right_training_graspable"
        / "hole_tol0p25mm_right_training_graspable.usd",
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
