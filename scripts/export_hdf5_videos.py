#!/usr/bin/env python3

"""
pip install h5py numpy imageio imageio-ffmpeg pillow

python export_hdf5_videos.py your_dataset.hdf5 ./exported_videos
"""


import argparse
import os

import h5py
import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:
    raise RuntimeError(
        "需要安装 imageio：pip install imageio imageio-ffmpeg"
    )

try:
    from PIL import Image
    import io
except ImportError:
    Image = None
    io = None


VIDEO_NAME_KEYS = [
    "video",
    "videos",
    "rgb",
    "camera",
    "image",
    "images",
    "color",
]


def looks_like_video(name, ds):
    lname = name.lower()
    shape = ds.shape

    # 名称判断
    name_match = any(k in lname for k in VIDEO_NAME_KEYS)

    # 4D 视频
    if ds.ndim == 4:
        # T,H,W,C
        if shape[-1] in (1, 3, 4):
            return True

        # T,C,H,W
        if shape[1] in (1, 3, 4):
            return True

    # 1D byte array / object array：
    # 有些数据集可能保存 JPEG/PNG bytes
    if ds.ndim == 1 and (name_match or ds.dtype.kind in "Ou"):
        return True

    return False


def convert_frame(frame):
    """
    把一帧转换为 imageio 可写入的视频格式。
    """

    # HDF5 scalar / object
    if isinstance(frame, np.ndarray):
        pass

    # JPEG / PNG encoded bytes
    if isinstance(frame, (bytes, bytearray)):
        if Image is None:
            raise RuntimeError(
                "检测到编码图片 bytes，但 Pillow 未安装："
                "pip install pillow"
            )

        img = Image.open(io.BytesIO(frame)).convert("RGB")
        return np.asarray(img)

    frame = np.asarray(frame)

    # uint8 视频直接使用
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            # 0~1
            if frame.max() <= 1.0:
                frame = frame * 255.0

            frame = np.clip(frame, 0, 255).astype(np.uint8)

        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

    # C,H,W -> H,W,C
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    # 单通道 H,W,1 -> H,W
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]

    # RGBA
    if frame.ndim == 3 and frame.shape[-1] == 4:
        # 保持 RGB，避免某些编码器对 alpha 处理异常
        frame = frame[..., :3]

    return frame


def export_video_dataset(ds, output_path, fps=30):
    """
    导出单个视频 dataset。
    """

    print(f"\nExporting:")
    print(f"  HDF5 shape : {ds.shape}")
    print(f"  dtype      : {ds.dtype}")
    print(f"  output     : {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 情况 1：常见 T,H,W,C / T,C,H,W
    if ds.ndim == 4:
        frames = ds

        with imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            quality=8,
        ) as writer:

            for i in range(frames.shape[0]):
                frame = convert_frame(frames[i])
                writer.append_data(frame)

        print(f"  frames     : {frames.shape[0]}")
        return

    # 情况 2：一维 encoded image bytes
    if ds.ndim == 1:
        with imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            quality=8,
        ) as writer:

            for i in range(len(ds)):
                frame = ds[i]

                if isinstance(frame, np.ndarray) and frame.dtype == np.uint8:
                    frame = frame.tobytes()

                frame = convert_frame(frame)
                writer.append_data(frame)

        print(f"  frames     : {len(ds)}")
        return

    raise RuntimeError(
        f"不支持的视频 shape: {ds.shape}, dtype={ds.dtype}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "hdf5",
        help="输入 HDF5 文件"
    )

    parser.add_argument(
        "output_dir",
        help="视频输出目录"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="导出 FPS，默认 30"
    )

    args = parser.parse_args()

    print("=" * 100)
    print(f"Input : {args.hdf5}")
    print(f"Output: {args.output_dir}")
    print(f"FPS   : {args.fps}")
    print("=" * 100)

    exported = 0

    with h5py.File(args.hdf5, "r") as h5:

        def visitor(name, obj):
            nonlocal exported

            if not isinstance(obj, h5py.Dataset):
                return

            if not looks_like_video(name, obj):
                return

            # HDF5:
            # /data/demo_0/videos/head
            #
            # 输出：
            # output_dir/data/demo_0/videos/head.mp4

            relative_path = name.lstrip("/") + ".mp4"
            output_path = os.path.join(
                args.output_dir,
                relative_path,
            )

            try:
                export_video_dataset(
                    obj,
                    output_path,
                    fps=args.fps,
                )
                exported += 1

            except Exception as e:
                print(f"  FAILED: {name}")
                print(f"  ERROR : {e}")

        h5.visititems(visitor)

    print("\n" + "=" * 100)
    print(f"完成，共导出 {exported} 个视频")
    print("=" * 100)


if __name__ == "__main__":
    main()