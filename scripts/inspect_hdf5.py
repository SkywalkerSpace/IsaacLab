#!/usr/bin/env python3

"""
时间长度：176 steps

Action
Action
├── left_arm   7
├── right_arm  7
├── left_hand 11
└── right_hand11
└── 36D

Robot
├── joint_position      54D
├── joint_velocity      54D
├── root_pose            7D
├── root_velocity        6D
└── links_state       55 × 13D

Observation
├── hand_joint_state    22D
├── head_joint_state     3D
├── left_eef              7D
├── right_eef             7D
├── object               13D
└── 3 × RGB camera
    ├── left_wrist   256×256×3
    ├── right_wrist  256×256×3
    └── robot_pov    256×256×3


python inspect_hdf5.py your_dataset.hdf5
"""

import argparse
import h5py
import numpy as np


def format_bytes(n):
    """把字节数转换成易读格式。"""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    n = float(n)

    for unit in units:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024

    return f"{n:.2f} PB"


def print_attrs(obj, indent=""):
    """打印 HDF5 对象的 attributes。"""
    if not obj.attrs:
        return

    print(f"{indent}attrs:")

    for key, value in obj.attrs.items():
        try:
            if isinstance(value, np.ndarray):
                if value.size > 20:
                    display = (
                        f"ndarray shape={value.shape}, "
                        f"dtype={value.dtype}"
                    )
                else:
                    display = repr(value)
            else:
                display = repr(value)

        except Exception:
            display = f"<{type(value).__name__}>"

        print(f"{indent}  - {key}: {display}")


def classify_dataset(path, ds):
    """
    根据 Dataset 名称和 shape，对数据进行简单分类。
    """
    name = path.lower()
    shape = ds.shape
    ndim = ds.ndim

    # 视频 / 图像
    if any(
        x in name
        for x in [
            "video",
            "videos",
            "rgb",
            "camera",
            "image",
            "images",
            "color",
        ]
    ):
        return "VIDEO/IMAGE?"

    # Action
    if any(
        x in name
        for x in [
            "action",
            "actions",
            "command",
            "commands",
        ]
    ):
        return "ACTION"

    # State / Observation
    if any(
        x in name
        for x in [
            "state",
            "states",
            "observation",
            "observations",
            "proprio",
        ]
    ):
        return "STATE/OBS"

    # Environment
    if any(
        x in name
        for x in [
            "env",
            "environment",
            "scene",
            "task",
        ]
    ):
        return "ENVIRONMENT"

    # 根据 shape 猜测视频
    if ndim == 4:
        # T,H,W,C
        if shape[-1] in (1, 3, 4):
            return "VIDEO/IMAGE?"

        # T,C,H,W
        if shape[1] in (1, 3, 4):
            return "VIDEO/IMAGE?"

    # 常见向量
    if ndim == 2 and min(shape) <= 128:
        return "VECTOR?"

    return ""


def print_dataset_info(path, ds, indent=""):
    """打印单个 Dataset 的详细信息。"""
    size_bytes = ds.size * ds.dtype.itemsize if ds.size else 0
    category = classify_dataset(path, ds)

    print(
        f"{indent}[DATASET] {path}\n"
        f"{indent}  shape       = {ds.shape}\n"
        f"{indent}  ndim        = {ds.ndim}\n"
        f"{indent}  dtype       = {ds.dtype}\n"
        f"{indent}  size        = {ds.size:,} elements\n"
        f"{indent}  raw_size    = {format_bytes(size_bytes)}"
    )

    if ds.chunks is not None:
        print(f"{indent}  chunks      = {ds.chunks}")

    if ds.compression is not None:
        print(
            f"{indent}  compression = {ds.compression}, "
            f"level={ds.compression_opts}"
        )

    if category:
        print(f"{indent}  category    = {category}")

    # 只读取少量样本，避免把整个视频加载进内存
    try:
        if ds.size > 0:

            if ds.ndim == 0:
                value = ds[()]

            elif ds.ndim == 1:
                value = ds[: min(10, ds.shape[0])]

            else:
                slices = tuple(
                    slice(0, 1)
                    for _ in ds.shape
                )
                value = ds[slices]

            print(
                f"{indent}  sample      = "
                f"{repr(value)}"
            )

    except Exception as e:
        print(
            f"{indent}  sample      = "
            f"<read failed: {e}>"
        )

    print_attrs(ds, indent + "  ")


def print_tree(group, depth=0, parent_path=""):
    """
    递归打印完整 HDF5 树结构。
    """
    indent = "  " * depth

    for name, obj in group.items():

        path = (
            f"{parent_path}/{name}"
            if parent_path
            else f"/{name}"
        )

        if isinstance(obj, h5py.Group):

            print(f"{indent}[GROUP]   {path}")

            print_attrs(
                obj,
                indent + "  "
            )

            print_tree(
                obj,
                depth + 1,
                path
            )

        elif isinstance(obj, h5py.Dataset):

            print_dataset_info(
                path,
                obj,
                indent
            )


def collect_datasets(h5):
    """
    收集所有 Dataset。
    """
    datasets = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj))

    h5.visititems(visitor)

    return datasets


def analyze_special_data(h5):
    """
    对 Action、State、Video、Environment、
    Meta 等数据做集中分析。
    """

    datasets = collect_datasets(h5)

    print("\n" + "=" * 100)
    print("重点数据分析")
    print("=" * 100)

    # =========================================================
    # 1. ACTION
    # =========================================================

    print("\n[1] ACTION DATA")

    found = False

    for name, ds in datasets:

        lname = name.lower()

        if any(
            k in lname
            for k in [
                "action",
                "actions",
            ]
        ):
            found = True

            print(f"  {name}")
            print(f"    shape = {ds.shape}")
            print(f"    ndim  = {ds.ndim}")
            print(f"    dtype = {ds.dtype}")

    if not found:
        print("  未发现明显的 action 数据集")

    # =========================================================
    # 2. STATE
    # =========================================================

    print("\n[2] STATE / OBSERVATION DATA")

    found = False

    for name, ds in datasets:

        lname = name.lower()

        if any(
            k in lname
            for k in [
                "state",
                "states",
                "observation",
                "observations",
                "proprio",
            ]
        ):
            found = True

            print(f"  {name}")
            print(f"    shape = {ds.shape}")
            print(f"    ndim  = {ds.ndim}")
            print(f"    dtype = {ds.dtype}")

    if not found:
        print(
            "  未发现明显的 "
            "state / observation 数据集"
        )

    # =========================================================
    # 3. VIDEO
    # =========================================================

    print("\n[3] VIDEO / IMAGE DATA")

    found = False

    for name, ds in datasets:

        lname = name.lower()
        shape = ds.shape

        maybe_video = any(
            k in lname
            for k in [
                "video",
                "videos",
                "rgb",
                "camera",
                "image",
                "images",
                "color",
            ]
        )

        # T,H,W,C 或 T,C,H,W
        if ds.ndim == 4:

            if (
                shape[-1]
                in (1, 3, 4)
            ):
                maybe_video = True

            elif (
                shape[1]
                in (1, 3, 4)
            ):
                maybe_video = True

        if not maybe_video:
            continue

        found = True

        print(f"  {name}")
        print(f"    shape = {shape}")
        print(f"    ndim  = {ds.ndim}")
        print(f"    dtype = {ds.dtype}")

        if ds.ndim >= 4:

            # T,H,W,C
            if shape[-1] in (1, 3, 4):

                print(
                    "    推测格式 = T,H,W,C"
                )

                print(
                    f"    帧数 = {shape[0]}"
                )

                print(
                    f"    分辨率 = "
                    f"{shape[2]}x{shape[1]}"
                )

                print(
                    f"    channels = {shape[3]}"
                )

            # T,C,H,W
            elif shape[1] in (1, 3, 4):

                print(
                    "    推测格式 = T,C,H,W"
                )

                print(
                    f"    帧数 = {shape[0]}"
                )

                print(
                    f"    分辨率 = "
                    f"{shape[3]}x{shape[2]}"
                )

                print(
                    f"    channels = {shape[1]}"
                )

    if not found:
        print(
            "  未发现明显的视频/图像数据"
        )

    # =========================================================
    # 4. ENVIRONMENT / META
    # =========================================================

    print("\n[4] ENVIRONMENT / META")

    found = False

    for name, ds in datasets:

        lname = name.lower()

        if any(
            k in lname
            for k in [
                "env",
                "environment",
                "scene",
                "task",
                "episode",
                "meta",
                "metadata",
            ]
        ):

            found = True

            print(f"  {name}")
            print(f"    shape = {ds.shape}")
            print(f"    ndim  = {ds.ndim}")
            print(f"    dtype = {ds.dtype}")

    if not found:
        print(
            "  未发现明显的 "
            "environment / metadata 数据"
        )

    # =========================================================
    # 5. ROOT ATTRIBUTES
    # =========================================================

    print("\n[5] ROOT ATTRIBUTES")

    if h5.attrs:

        for key, value in h5.attrs.items():

            print(
                f"  {key}: "
                f"{repr(value)}"
            )

    else:

        print("  无 root attributes")


def show_all_dataset_shapes(h5):
    """
    集中显示所有 Dataset 的维度信息。
    """

    print("\n" + "=" * 100)
    print("所有 Dataset 维度")
    print("=" * 100)

    datasets = collect_datasets(h5)

    for name, ds in datasets:

        print(
            f"{name:70s} "
            f"shape={str(ds.shape):25s} "
            f"ndim={ds.ndim:<2} "
            f"dtype={ds.dtype}"
        )


def show_action_state_shapes(h5):
    """
    单独显示 Action / State 的 shape，
    方便检查时间维度是否 T / T+1。
    """

    print("\n" + "=" * 100)
    print("ACTION / STATE 维度汇总")
    print("=" * 100)

    datasets = collect_datasets(h5)

    action_items = []
    state_items = []

    for name, ds in datasets:

        lname = name.lower()

        if any(
            k in lname
            for k in [
                "action",
                "actions",
            ]
        ):
            action_items.append(
                (name, ds)
            )

        if any(
            k in lname
            for k in [
                "state",
                "states",
                "observation",
                "observations",
                "proprio",
            ]
        ):
            state_items.append(
                (name, ds)
            )

    # -------------------------
    # Action
    # -------------------------

    print("\nACTION:")

    if action_items:

        for name, ds in action_items:

            print(
                f"  {name}"
                f"  shape={ds.shape}"
                f"  dtype={ds.dtype}"
            )

    else:

        print(
            "  未发现 action 数据"
        )

    # -------------------------
    # State
    # -------------------------

    print("\nSTATE / OBSERVATION:")

    if state_items:

        for name, ds in state_items:

            print(
                f"  {name}"
                f"  shape={ds.shape}"
                f"  dtype={ds.dtype}"
            )

    else:

        print(
            "  未发现 state / observation 数据"
        )

    # -------------------------
    # 简单检查时间维度
    # -------------------------

    if action_items and state_items:

        print(
            "\n时间维度检查:"
        )

        for action_name, action_ds in action_items:

            if action_ds.ndim == 0:
                continue

            action_t = action_ds.shape[0]

            for state_name, state_ds in state_items:

                if state_ds.ndim == 0:
                    continue

                state_t = state_ds.shape[0]

                print(
                    f"  action: "
                    f"{action_name}"
                )

                print(
                    f"  state : "
                    f"{state_name}"
                )

                print(
                    f"    action T = {action_t}, "
                    f"state T = {state_t}"
                )

                if state_t == action_t:
                    print(
                        "    -> T 相同"
                    )

                elif state_t == action_t + 1:
                    print(
                        "    -> state = action + 1，"
                        "可能存在常见的 "
                        "s_t -> a_t -> s_(t+1) 对齐"
                    )

                elif action_t == state_t + 1:
                    print(
                        "    -> action = state + 1"
                    )

                else:
                    print(
                        "    -> 时间维度不同，"
                        "需要进一步检查"
                    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "查看 HDF5 完整结构及 "
            "action/state/video/environment 信息"
        )
    )

    parser.add_argument(
        "hdf5",
        help="HDF5 文件路径"
    )

    args = parser.parse_args()

    print("=" * 100)
    print(
        f"HDF5 FILE: {args.hdf5}"
    )
    print("=" * 100)

    with h5py.File(
        args.hdf5,
        "r"
    ) as h5:

        # =====================================================
        # 1. 完整树结构
        # =====================================================

        print("\n" + "=" * 100)
        print("完整 HDF5 树结构")
        print("=" * 100)

        print("[ROOT] /")

        print_attrs(
            h5,
            "  "
        )

        print_tree(h5)

        # =====================================================
        # 2. 所有 Dataset 的维度汇总
        # =====================================================

        show_all_dataset_shapes(h5)

        # =====================================================
        # 3. Action / State 专门汇总
        # =====================================================

        show_action_state_shapes(h5)

        # =====================================================
        # 4. Video / Environment / Meta 分析
        # =====================================================

        analyze_special_data(h5)


if __name__ == "__main__":
    main()
