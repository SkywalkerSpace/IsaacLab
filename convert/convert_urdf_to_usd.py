# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the robot URDFs in ``convert/assets/robots`` to standalone USDs.

Run from an Isaac Lab environment::

    ./isaaclab.sh -p convert/convert_robot_urdfs_to_usd.py

By default, each USD is written alongside its URDF. Use ``--output-dir`` to
write the two files to another directory.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_URDF_DIR = SCRIPT_DIR / "assets" / "robots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf-dir",
        type=Path,
        default=DEFAULT_URDF_DIR,
        help="Directory containing the URDF files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: same directory as --urdf-dir)",
    )
    parser.add_argument("--fix-base", action="store_true", help="Import each robot with a fixed base")
    parser.add_argument(
        "--merge-fixed-joints", action="store_true", help="Merge fixed joints during import"
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def convert_urdf(urdf_path: Path, output_path: Path, *, fix_base: bool, merge_fixed_joints: bool) -> None:
    """Convert one URDF into one flattened USD file."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from pxr import Usd

    urdf_path = urdf_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF does not exist: {urdf_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep importer intermediates and sidecars out of the final output folder.
    with tempfile.TemporaryDirectory(prefix=f"{urdf_path.stem}_") as temp_dir:
        converter = UrdfConverter(
            UrdfConverterCfg(
                asset_path=str(urdf_path),
                usd_dir=temp_dir,
                usd_file_name=output_path.name,
                fix_base=fix_base,
                merge_fixed_joints=merge_fixed_joints,
                force_usd_conversion=True,
                make_instanceable=False,
            )
        )
        converter_path = Path(converter.usd_path).resolve()
        stage = Usd.Stage.Open(str(converter_path), load=Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError(f"Could not open converted USD: {converter_path}")
        if not stage.Flatten().Export(str(output_path)):
            raise RuntimeError(f"Could not write flattened USD: {output_path}")

    print(f"[OK] {urdf_path} -> {output_path}")


def main() -> None:
    args = parse_args()
    urdf_dir = args.urdf_dir.expanduser().resolve()
    urdf_paths = sorted(urdf_dir.glob("*.urdf"))
    if not urdf_paths:
        raise FileNotFoundError(f"No URDF files found in {urdf_dir}")

    output_dir = (args.output_dir or urdf_dir).expanduser().resolve()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        for urdf_path in urdf_paths:
            convert_urdf(
                urdf_path,
                output_dir / f"{urdf_path.stem}.usd",
                fix_base=args.fix_base,
                merge_fixed_joints=args.merge_fixed_joints,
            )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
