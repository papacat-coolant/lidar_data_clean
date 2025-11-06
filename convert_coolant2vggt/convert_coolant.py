#!/usr/bin/env python3
"""
Convert Coolant lidar-matched dataset into the unified VGGt-ready format.

The script mirrors `convert_syndrone.py`, but it is tailored to Coolant
collection structure where each frame stores:

  - `images/<stem>.JPG`                      # RGB image (already EXIF oriented)
  - `raw_depths/<stem>_depth.npy`            # float32 z-depth in meters
  - `poses/<stem>_pose.npy`                  # dict with `K`, `R`, `T`, `camtoworld`

Output layout (matching convert_eden.py expectations):

  output_dir/
    images/00000.jpg ...                     # RGB images copied & renamed
    depths/00000.npy ...                     # float32 z-depth maps
    image_names.json                         # list of filenames
    cam_from_worlds.npy                      # (N,3,4) OpenCV world->camera matrices
    intrinsics.npy                           # (N,3,3) intrinsics per frame
    complete_log.txt                         # conversion summary

All poses are converted from camera-to-world (c2w) to world-to-camera (w2c)
in the OpenCV right-handed coordinate system. Depths are preserved as z-depth
maps (perpendicular distance along the optical axis).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


IMAGE_FOLDER_NAME = "images"
DEPTH_FOLDER_NAME = "depths"
COMPLETION_INDICATOR_FILE = "complete_log.txt"


def _ensure_dataset_root(path: Path) -> Path:
    """Return the directory that actually contains Coolant frame folders.

    Users sometimes pass the parent directory that wraps a single
    `{scene_name}/{scene_name}` nesting. If `images/` already exists under the
    provided path we keep it; otherwise we search one level deeper for a
    directory containing the expected sub-folders.
    """

    path = path.expanduser().resolve()
    if (path / "images").is_dir() and (path / "raw_depths").is_dir():
        return path

    candidates = [p for p in path.iterdir() if p.is_dir()]
    for candidate in candidates:
        if (candidate / "images").is_dir() and (candidate / "raw_depths").is_dir():
            return candidate

    raise ValueError(
        f"Cannot locate Coolant dataset structure under {path}. "
        "Expected `images/`, `raw_depths/`, and `poses/` sub-folders."
    )


def _list_images(images_dir: Path) -> List[Path]:
    """Return sorted list of image paths (supports common Coolant extensions)."""

    patterns = ["*.JPG", "*.JPEG", "*.jpg", "*.jpeg", "*.png"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(images_dir.glob(pattern))
    return sorted({f.resolve() for f in files})


def _load_pose_dict(pose_path: Path) -> dict:
    data = np.load(pose_path, allow_pickle=True)

    # Pose files are stored as pickled dictionaries (0-d object arrays).
    if isinstance(data, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected .npy dictionary, got .npz in {pose_path}")

    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        return data.item()

    raise ValueError(f"Unsupported pose format in {pose_path}: type={type(data)}")


def _c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    if c2w.shape == (3, 4):
        # Upgrade to 4x4 for inversion if needed.
        c2w = np.vstack([c2w, np.array([0.0, 0.0, 0.0, 1.0], dtype=c2w.dtype)])

    if c2w.shape != (4, 4):
        raise ValueError(f"c2w must be (4,4) or (3,4); received {c2w.shape}")

    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    return np.hstack([R_w2c, t_w2c.reshape(3, 1)])


def convert_coolant_scene(
    input_dir: Path,
    output_dir: Path,
    dataset_name: str = "coolant",
    version: str = "0.1",
    limit: int = 0,
) -> None:
    """Convert a Coolant scene directory into the unified format."""

    scene_dir = _ensure_dataset_root(input_dir)
    images_dir = scene_dir / "images"
    depths_dir = scene_dir / "raw_depths"
    poses_dir = scene_dir / "poses"

    if not poses_dir.is_dir():
        raise ValueError(f"Poses directory missing: {poses_dir}")

    image_files = _list_images(images_dir)
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    if limit > 0 and len(image_files) > limit:
        image_files = image_files[:limit]
        print(f"[INFO] Limiting to {limit} frames (out of {len(image_files)} total)")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_images = output_dir / IMAGE_FOLDER_NAME
    out_depths = output_dir / DEPTH_FOLDER_NAME
    out_images.mkdir(exist_ok=True)
    out_depths.mkdir(exist_ok=True)

    image_names: List[str] = []
    cam_from_worlds: List[np.ndarray] = []
    intrinsics: List[np.ndarray] = []

    for idx, image_path in enumerate(tqdm(image_files, desc="Converting frames")):
        stem = image_path.stem
        depth_path = depths_dir / f"{stem}_depth.npy"
        pose_path = poses_dir / f"{stem}_pose.npy"

        if not depth_path.is_file():
            print(f"[WARN] Missing depth file for {stem}, skipping frame.")
            continue
        if not pose_path.is_file():
            print(f"[WARN] Missing pose file for {stem}, skipping frame.")
            continue

        try:
            pose_dict = _load_pose_dict(pose_path)
        except Exception as exc:  # pragma: no cover
            print(f"[ERROR] Failed to load pose {pose_path.name}: {exc}")
            continue

        K = np.asarray(pose_dict.get("K"), dtype=np.float64)
        c2w_raw = pose_dict.get("camtoworld", None)
        if c2w_raw is None:
            c2w_raw = pose_dict.get("c2w", None)
        if c2w_raw is None:
            print(f"[ERROR] Pose dict missing 'camtoworld'/'c2w' for {stem}")
            continue
        c2w = np.asarray(c2w_raw, dtype=np.float64)

        if K.shape != (3, 3):
            print(f"[ERROR] Intrinsics must be 3x3 for {stem}, got {K.shape}")
            continue
        if c2w.shape not in ((3, 4), (4, 4)):
            print(f"[ERROR] camtoworld must be 3x4 or 4x4 for {stem}, got {c2w.shape}")
            continue

        depth = np.load(depth_path)
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2:
            print(f"[ERROR] Depth map must be 2D for {stem}, got shape {depth.shape}")
            continue

        try:
            w2c = _c2w_to_w2c(c2w)
        except Exception as exc:
            print(f"[ERROR] Failed to convert pose for {stem}: {exc}")
            continue

        frame_id = f"{len(image_names):05d}"
        out_rgb_name = f"{frame_id}.jpg"
        out_depth_name = f"{frame_id}.npy"

        out_rgb_path = out_images / out_rgb_name
        out_depth_path = out_depths / out_depth_name

        try:
            shutil.copy2(image_path, out_rgb_path)
        except Exception as exc:  # pragma: no cover
            print(f"[ERROR] Failed to copy RGB for {stem}: {exc}")
            continue

        try:
            np.save(out_depth_path, depth)
        except Exception as exc:
            print(f"[ERROR] Failed to save depth for {stem}: {exc}")
            out_rgb_path.unlink(missing_ok=True)
            continue

        image_names.append(out_rgb_name)
        cam_from_worlds.append(w2c)
        intrinsics.append(K)

    if not image_names:
        raise RuntimeError("No frames were successfully converted.")

    cam_from_worlds_array = np.stack(cam_from_worlds).astype(np.float64)
    intrinsics_array = np.stack(intrinsics).astype(np.float64)

    dets = np.linalg.det(cam_from_worlds_array[:, :, :3])
    if not np.allclose(dets, 1.0, atol=1e-2):
        print(
            "[WARN] Rotation determinants deviate from 1.0: "
            f"[{dets.min():.4f}, {dets.max():.4f}]"
        )

    with open(output_dir / "image_names.json", "w", encoding="utf-8") as f:
        json.dump(image_names, f, indent=2)

    np.save(output_dir / "cam_from_worlds.npy", cam_from_worlds_array)
    np.save(output_dir / "intrinsics.npy", intrinsics_array)

    completion_path = output_dir / COMPLETION_INDICATOR_FILE
    with completion_path.open("w", encoding="utf-8") as f:
        f.write(f"Conversion completed at: {datetime.now().isoformat()}\n")
        f.write(f"Input directory: {scene_dir}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Version: {version}\n")
        f.write(f"Frames converted: {len(image_names)}\n")
        f.write("\n")
        f.write("Output verification:\n")
        f.write(
            f"  - cam_from_worlds shape: {cam_from_worlds_array.shape}\n"
        )
        f.write(f"  - intrinsics shape: {intrinsics_array.shape}\n")
        f.write(f"  - Rotation det range: [{dets.min():.6f}, {dets.max():.6f}]\n")

    print(f"[SUCCESS] Converted {len(image_names)} frames to {output_dir}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Coolant dataset to unified VGGt-ready format"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to Coolant scene (contains images/, raw_depths/, poses/)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: <input_dir>_converted)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="coolant",
        help="Dataset name stored in the completion log",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="0.1",
        help="Dataset version stored in the completion log",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of frames to convert (0 = all)",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(str(args.input_dir).rstrip("/") + "_converted")

    print(f"[INFO] Input:  {args.input_dir}")
    print(f"[INFO] Output: {output_dir}")

    convert_coolant_scene(
        input_dir=args.input_dir,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        version=args.version,
        limit=args.limit,
    )

    print("[INFO] Conversion complete.")
    print(
        "[INFO] You can visualize the converted results with "
        "`coolant_multi_frame_vis.py --data_dir <output_dir>`"
    )


if __name__ == "__main__":
    main()


