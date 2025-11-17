#!/usr/bin/env python3
"""
Convert Synthetic Coolant dataset to VGGT unified format.

Input format (Synthetic):
  - images/*.jpeg                         # RGB images (LS_Coolant_Color.XXXX.jpeg)
  - metadata/*.exr                        # Depth maps (LS_Coolant_metadata.FinalImage_depth.XXXX.exr)
  - poses/*.npy                          # Pose dictionaries (pose.XXXX.npy)

Pose dictionary structure:
  - K: (3,3) camera intrinsics matrix
  - R: (3,3) rotation matrix (world-to-camera)
  - T: (3,) translation vector (world-to-camera)
  - camtoworld: (4,4) camera-to-world transformation matrix

Output format (unified VGGT format):
  - images/00000.jpg, 00001.jpg, ...      # RGB images
  - depths/00000.npy, 00001.npy, ...      # z-depth in NPY format
  - image_names.json                      # list of image names
  - cam_from_worlds.npy                   # (N,3,4) world->camera w2c matrices
  - intrinsics.npy                        # (N,3,3) camera intrinsics
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
import cv2
from tqdm import tqdm

try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE = True
except ImportError:
    OPENEXR_AVAILABLE = False
    print("[WARN] OpenEXR not available. Install with: pip install OpenEXR")

IMAGE_FOLDER_NAME = "images"
DEPTH_FOLDER_NAME = "depths"
COMPLETION_INDICATOR_FILE = "complete_log.txt"


def read_exr_depth(exr_path: str) -> np.ndarray:
    """
    Read depth from EXR file.
    
    Args:
        exr_path: Path to EXR depth file
        
    Returns:
        depth: (H, W) float32 depth map
    """
    if not OPENEXR_AVAILABLE:
        raise ImportError("OpenEXR library not available. Install with: pip install OpenEXR")
    
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    
    # Get dimensions
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Read R channel (depth is stored in RGB channels, they're all the same)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = exr_file.channel('R', pt)
    
    # Convert to numpy array
    depth = np.frombuffer(depth_str, dtype=np.float32).reshape(height, width)
    
    return depth


def load_synthetic_frame(
    rgb_path: str,
    depth_path: str,
    pose_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one frame from Synthetic dataset.
    
    Args:
        rgb_path: Path to RGB image
        depth_path: Path to depth EXR file
        pose_path: Path to pose NPY file
        
    Returns:
        rgb: (H, W, 3) RGB image (uint8)
        depth: (H, W) depth map (float32)
        K: (3, 3) intrinsics matrix
        w2c: (3, 4) world-to-camera transformation matrix
    """
    # Load RGB
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None:
        raise ValueError(f"Cannot read RGB: {rgb_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    
    # Load depth from EXR
    depth = read_exr_depth(depth_path)
    
    # Load pose dictionary
    pose_dict = np.load(pose_path, allow_pickle=True).item()
    
    # Extract intrinsics
    K = pose_dict['K'].astype(np.float64)
    
    # Extract world-to-camera transformation
    # The pose dict contains R (rotation) and T (translation) which are w2c
    R_w2c = pose_dict['R'].astype(np.float64)
    t_w2c = pose_dict['T'].astype(np.float64)
    
    # Build 3x4 w2c matrix [R | t]
    w2c = np.hstack([R_w2c, t_w2c.reshape(3, 1)])
    
    # Verify shapes
    assert rgb.shape[:2] == depth.shape, \
        f"RGB and depth dimensions mismatch: {rgb.shape[:2]} vs {depth.shape}"
    
    return rgb, depth, K, w2c


def convert_synthetic_scene(
    input_dir: Path,
    output_dir: Path,
    dataset_name: str = "synthetic_coolant",
    version: str = "0.1",
    limit: int = 0,
    depth_scale: float = 1.0
):
    """
    Convert one Synthetic scene to unified VGGT format.
    
    Args:
        input_dir: Path to scene directory (contains images/, metadata/, poses/)
        output_dir: Path to output directory
        dataset_name: Dataset name for metadata
        version: Dataset version for metadata
        limit: Maximum number of frames to convert (0 = all)
        depth_scale: Depth scaling factor to apply (default: 1.0, no scaling)
    """
    # Check if OpenEXR is available
    if not OPENEXR_AVAILABLE:
        raise ImportError(
            "OpenEXR library is required to read depth files. "
            "Install with: pip install OpenEXR"
        )
    
    # Create output directories
    out_images = output_dir / IMAGE_FOLDER_NAME
    out_depths = output_dir / DEPTH_FOLDER_NAME
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_depths, exist_ok=True)
    
    # Find all pose files (these are the authoritative source for frame list)
    pose_dir = input_dir / "poses"
    pose_files = sorted(glob.glob(str(pose_dir / "pose.*.npy")))
    
    if not pose_files:
        raise ValueError(f"No pose files found in {pose_dir}")
    
    # Apply limit if specified
    if limit > 0 and len(pose_files) > limit:
        pose_files = pose_files[:limit]
        print(f"[INFO] Limiting to {limit} frames (out of total available)")
    
    print(f"[INFO] Found {len(pose_files)} frames in {input_dir}")
    
    # Prepare data lists
    image_names = []
    cam_from_worlds_list = []
    intrinsics_list = []
    
    # Process each frame
    for idx, pose_path in enumerate(tqdm(pose_files, desc="Converting frames")):
        # Extract frame number from filename (e.g., pose.0123.npy -> 0123)
        frame_num = Path(pose_path).stem.split('.')[-1]
        
        # Build corresponding file paths
        rgb_path = str(input_dir / "images" / f"LS_Coolant_Color.{frame_num}.jpeg")
        depth_path = str(input_dir / "metadata" / f"LS_Coolant_metadata.FinalImage_depth.{frame_num}.exr")
        
        # Check if corresponding files exist
        if not os.path.exists(rgb_path):
            print(f"[WARN] Missing RGB for frame {frame_num}, skipping")
            continue
        if not os.path.exists(depth_path):
            print(f"[WARN] Missing depth for frame {frame_num}, skipping")
            continue
        
        try:
            # Load frame data
            rgb, depth, K, w2c = load_synthetic_frame(rgb_path, depth_path, pose_path)
            
            # Apply depth scaling if specified
            if depth_scale != 1.0:
                depth = depth * depth_scale
            
            # Generate output filename
            frame_id = f"{idx:05d}"
            out_rgb_name = f"{frame_id}.jpg"
            out_depth_name = f"{frame_id}.npy"
            
            # Save RGB
            out_rgb_path = str(out_images / out_rgb_name)
            cv2.imwrite(out_rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            # Save depth as NPY
            out_depth_path = str(out_depths / out_depth_name)
            np.save(out_depth_path, depth.astype(np.float32))
            
            # Store metadata
            image_names.append(out_rgb_name)
            cam_from_worlds_list.append(w2c)
            intrinsics_list.append(K)
            
        except Exception as e:
            print(f"[ERROR] Failed to process frame {frame_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not image_names:
        raise RuntimeError(f"No frames successfully converted from {input_dir}")
    
    # Convert to numpy arrays
    cam_from_worlds_array = np.array(cam_from_worlds_list, dtype=np.float64)  # (N, 3, 4)
    intrinsics_array = np.array(intrinsics_list, dtype=np.float64)  # (N, 3, 3)
    
    # Verify output format
    n_frames = len(image_names)
    assert cam_from_worlds_array.shape == (n_frames, 3, 4), \
        f"cam_from_worlds must be (N,3,4), got {cam_from_worlds_array.shape}"
    assert intrinsics_array.shape == (n_frames, 3, 3), \
        f"intrinsics must be (N,3,3), got {intrinsics_array.shape}"
    
    # Verify w2c matrices (rotation part should have det ≈ 1 or -1)
    R_w2c = cam_from_worlds_array[:, :, :3]  # (N, 3, 3)
    dets = np.linalg.det(R_w2c)
    assert np.allclose(np.abs(dets), 1.0, atol=1e-2), \
        f"Rotation matrices should have |det| ≈ 1, got range [{dets.min():.4f}, {dets.max():.4f}]"
    
    print(f"[INFO] Output format verified:")
    print(f"       - cam_from_worlds: {cam_from_worlds_array.shape} (world->camera w2c)")
    print(f"       - intrinsics: {intrinsics_array.shape}")
    print(f"       - Rotation determinants: [{dets.min():.4f}, {dets.max():.4f}]")
    
    # Save metadata files
    with open(output_dir / "image_names.json", 'w') as f:
        json.dump(image_names, f, indent=2)
    
    np.save(output_dir / "cam_from_worlds.npy", cam_from_worlds_array)
    np.save(output_dir / "intrinsics.npy", intrinsics_array)
    
    # Create completion indicator
    completion_path = output_dir / COMPLETION_INDICATOR_FILE
    with open(completion_path, 'w') as f:
        f.write(f"Conversion completed at: {datetime.now().isoformat()}\n")
        f.write(f"Input: {input_dir}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Version: {version}\n")
        f.write(f"Frames: {len(image_names)}\n")
        f.write(f"Depth scale: {depth_scale}\n")
        f.write(f"\n")
        f.write(f"Output format verification:\n")
        f.write(f"  - Depth: z-depth format\n")
        f.write(f"  - Pose: world-to-camera (w2c) transformation\n")
        f.write(f"  - cam_from_worlds shape: {cam_from_worlds_array.shape}\n")
        f.write(f"  - intrinsics shape: {intrinsics_array.shape}\n")
    
    print(f"[SUCCESS] Converted {len(image_names)} frames to {output_dir}")
    print(f"[SUCCESS] ✓ Depth: z-depth format")
    print(f"[SUCCESS] ✓ Pose: world-to-camera (w2c)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert Synthetic Coolant dataset to VGGT unified format"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing images/, metadata/, poses/ subdirectories"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: input_dir_vggt)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="synthetic_coolant",
        help="Dataset name for metadata"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="0.1",
        help="Dataset version for metadata"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of frames to convert (0 = all frames)"
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1.0,
        help="Depth scaling factor (default: 1.0, no scaling)"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = Path(str(args.input_dir).rstrip('/') + "_vggt")
    
    print(f"[INFO] Input:  {args.input_dir}")
    print(f"[INFO] Output: {args.output_dir}")
    print(f"[INFO] Depth scale: {args.depth_scale}")
    
    # Convert
    convert_synthetic_scene(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        version=args.version,
        limit=args.limit,
        depth_scale=args.depth_scale
    )
    
    print("[INFO] Conversion complete!")
    print(f"[INFO] You can now visualize with:")
    print(f"       python vis_output_test.py --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()

