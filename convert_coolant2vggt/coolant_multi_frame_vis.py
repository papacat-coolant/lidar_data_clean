#!/usr/bin/env python3
"""
Visualize Coolant dataset frames (raw or converted) with nerfvis.

The viewer renders camera frustums, RGB billboards, and depth-derived
point clouds. It supports two input layouts:

1. Raw Coolant scene (images/, raw_depths/, poses/)
2. Converted scene produced by convert_coolant.py (images/, depths/, cam_from_worlds.npy)

Usage examples:

    python coolant_multi_frame_vis.py --data_dir /path/to/raw/scene --limit 50
    python coolant_multi_frame_vis.py --data_dir /path/to/converted_scene --stride 4
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import nerfvis
import numpy as np
from tqdm import tqdm


@dataclass
class FrameData:
    name: str
    image_path: Path
    depth_path: Path
    K: np.ndarray
    c2w: np.ndarray


def _detect_layout(data_dir: Path) -> str:
    if (data_dir / "cam_from_worlds.npy").is_file():
        return "converted"
    if (data_dir / "poses").is_dir() and (data_dir / "raw_depths").is_dir():
        return "raw"
    raise ValueError(
        f"Unrecognised Coolant layout under {data_dir}. "
        "Expected converted outputs or raw scene folders."
    )


def _invert_w2c(w2c: np.ndarray) -> np.ndarray:
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    c2w = np.eye(4, dtype=w2c.dtype)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def _load_frames_converted(data_dir: Path, limit: int, frame_stride: int) -> List[FrameData]:
    images_dir = data_dir / "images"
    depths_dir = data_dir / "depths"

    with open(data_dir / "image_names.json", "r", encoding="utf-8") as f:
        image_names: List[str] = json.load(f)

    cam_from_worlds = np.load(data_dir / "cam_from_worlds.npy")
    intrinsics = np.load(data_dir / "intrinsics.npy")

    if frame_stride > 1:
        image_names = image_names[::frame_stride]
        cam_from_worlds = cam_from_worlds[::frame_stride]
        intrinsics = intrinsics[::frame_stride]

    if limit > 0:
        image_names = image_names[:limit]
        cam_from_worlds = cam_from_worlds[:limit]
        intrinsics = intrinsics[:limit]

    frames: List[FrameData] = []
    for idx, name in enumerate(image_names):
        depth_name = Path(name).with_suffix(".npy").name
        frames.append(
            FrameData(
                name=name,
                image_path=images_dir / name,
                depth_path=depths_dir / depth_name,
                K=np.asarray(intrinsics[idx], dtype=np.float64),
                c2w=_invert_w2c(np.asarray(cam_from_worlds[idx], dtype=np.float64)),
            )
        )
    return frames


def _load_pose_dict(pose_path: Path) -> dict:
    data = np.load(pose_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        return data.item()
    raise ValueError(f"Unsupported pose format in {pose_path}")


def _c2w_from_pose_dict(pose_dict: dict) -> np.ndarray:
    c2w = np.asarray(
        pose_dict.get("camtoworld") or pose_dict.get("c2w"),
        dtype=np.float64,
    )
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)])
    if c2w.shape != (4, 4):
        raise ValueError(f"Pose matrix must be 3x4 or 4x4, got {c2w.shape}")
    return c2w


def _load_frames_raw(data_dir: Path, limit: int, frame_stride: int) -> List[FrameData]:
    images_dir = data_dir / "images"
    depths_dir = data_dir / "raw_depths"
    poses_dir = data_dir / "poses"

    patterns = ["*.JPG", "*.JPEG", "*.jpg", "*.jpeg", "*.png"]
    image_paths: List[Path] = []
    for pattern in patterns:
        image_paths.extend(images_dir.glob(pattern))
    image_paths = sorted({p.resolve() for p in image_paths})

    if frame_stride > 1:
        image_paths = image_paths[::frame_stride]

    if limit > 0:
        image_paths = image_paths[:limit]

    frames: List[FrameData] = []
    for image_path in image_paths:
        stem = image_path.stem
        depth_path = depths_dir / f"{stem}_depth.npy"
        pose_path = poses_dir / f"{stem}_pose.npy"
        if not depth_path.is_file() or not pose_path.is_file():
            continue
        pose_dict = _load_pose_dict(pose_path)
        K = np.asarray(pose_dict["K"], dtype=np.float64)
        frames.append(
            FrameData(
                name=image_path.name,
                image_path=image_path,
                depth_path=depth_path,
                K=K,
                c2w=_c2w_from_pose_dict(pose_dict),
            )
        )
    return frames


def _backproject_z_depth(
    depth: np.ndarray,
    K: np.ndarray,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    us = np.arange(0, W, stride)
    vs = np.arange(0, H, stride)
    uu, vv = np.meshgrid(us, vs)

    sampled_depth = depth[vv, uu]
    valid = sampled_depth > 0
    uu = uu[valid]
    vv = vv[valid]
    sampled_depth = sampled_depth[valid]

    x = (uu - cx) / fx * sampled_depth
    y = (vv - cy) / fy * sampled_depth
    pts_cam = np.stack([x, y, sampled_depth], axis=-1)
    return pts_cam, uu, vv


def _load_depth(path: Path) -> np.ndarray:
    depth = np.load(path)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be 2D at {path}")
    return depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=4, help="Depth sampling stride")
    parser.add_argument("--point_size", type=float, default=1.5)
    parser.add_argument("--keep_pct", type=float, default=100.0)
    parser.add_argument("--min_depth", type=float, default=0.0)
    parser.add_argument("--max_depth", type=float, default=0.0)
    parser.add_argument(
        "--global_extrinsic",
        type=Path,
        default=None,
        help="Optional 4x4 transform that maps dataset world into global coordinates",
    )
    parser.add_argument("--center_global", action="store_true")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Sample every Nth frame for faster visualization",
    )
    parser.add_argument(
        "--frustum_depth",
        type=float,
        default=20.0,
        help="Depth extent for camera frustums and billboards",
    )
    args = parser.parse_args()

    layout = _detect_layout(args.data_dir)
    if layout == "converted":
        frames = _load_frames_converted(args.data_dir, args.limit, max(1, args.frame_stride))
    else:
        frames = _load_frames_raw(args.data_dir, args.limit, max(1, args.frame_stride))

    if not frames:
        raise SystemExit("No frames found to visualize.")

    if args.global_extrinsic is not None:
        T_global = np.asarray(np.load(args.global_extrinsic), dtype=np.float64)
        if T_global.shape != (4, 4):
            raise ValueError("Global extrinsic must be 4x4")
    else:
        T_global = np.eye(4, dtype=np.float64)

    scene = nerfvis.Scene("Coolant Scene Viewer", default_opencv=True)
    scene.set_opencv()
    scene.set_opencv_world()

    all_points: List[np.ndarray] = []

    for idx, frame in enumerate(tqdm(frames, desc="Loading depths")):
        rgb_bgr = cv2.imread(str(frame.image_path), cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None:
            print(f"[WARN] Unable to read image {frame.image_path}, skipping.")
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth = _load_depth(frame.depth_path)

        pts_cam, uu, vv = _backproject_z_depth(depth, frame.K, max(1, args.stride))

        if pts_cam.size == 0:
            continue

        if args.keep_pct < 100.0 and pts_cam.shape[0] > 0:
            threshold = np.percentile(pts_cam[:, 2], args.keep_pct)
            keep = pts_cam[:, 2] <= threshold
            pts_cam = pts_cam[keep]
            uu = uu[keep]
            vv = vv[keep]
 
        if pts_cam.size == 0:
            continue

        if args.min_depth > 0:
            keep = pts_cam[:, 2] >= args.min_depth
            pts_cam = pts_cam[keep]
            uu = uu[keep]
            vv = vv[keep]

        if pts_cam.size == 0:
            continue

        if args.max_depth > 0:
            keep = pts_cam[:, 2] <= args.max_depth
            pts_cam = pts_cam[keep]
            uu = uu[keep]
            vv = vv[keep]

        if pts_cam.size == 0:
            continue

        c2w = frame.c2w.copy()
        pts_world = (c2w[:3, :3] @ pts_cam.T + c2w[:3, 3:4]).T

        if args.global_extrinsic is not None:
            pts_world_h = np.concatenate([pts_world, np.ones((len(pts_world), 1))], axis=1)
            pts_world = (T_global[:3, :] @ pts_world_h.T).T
            c2w = T_global @ c2w

        colors = (rgb[vv, uu] / 255.0).reshape(-1, 3)
        all_points.append(pts_world)

        frame_id = f"frame_{idx:04d}_{frame.name}"
        H, W = rgb.shape[:2]
        fx = frame.K[0, 0]

        z_depth = max(args.frustum_depth, 1e-3)

        scene.add_camera_frustum(
            f"camera/{frame_id}/frustum",
            r=c2w[:3, :3],
            t=c2w[:3, 3],
            focal_length=float(fx),
            image_width=W,
            image_height=H,
            z=float(z_depth),
        )
        scene.add_image(
            f"camera/{frame_id}/image",
            rgb,
            r=c2w[:3, :3],
            t=c2w[:3, 3],
            focal_length=float(fx),
            z=float(z_depth),
            image_size=min(1024, max(W, H)),
        )
        scene.add_points(
            f"points/{frame_id}",
            pts_world,
            point_size=args.point_size,
            vert_color=colors,
        )

    if args.center_global and all_points:
        stacked = np.vstack(all_points)
        mean = stacked.mean(axis=0)
        for name, node in list(scene.scene.nodes.items()):
            if name.startswith("points/"):
                node.points -= mean

    scene.add_axes()
    scene.display()


if __name__ == "__main__":
    main()


