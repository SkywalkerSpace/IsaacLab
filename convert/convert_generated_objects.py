"""Convert the procedural robosuite objects to standalone USD assets.

Run with Isaac Lab's Python launcher from the Isaac Lab repository, for example:

    ./isaaclab.sh -p /path/to/IsaacLab/collect/convert_generated_objects.py

Outputs are written to ``collect/assets/mjcf/generated/``. For the model list
and conversion details, see :mod:`convert_mjcf_to_usd`.
"""

from __future__ import annotations

import sys

import convert_mjcf_to_usd


if __name__ == "__main__":
    sys.argv.append("--generated")
    raise SystemExit(convert_mjcf_to_usd.main())
