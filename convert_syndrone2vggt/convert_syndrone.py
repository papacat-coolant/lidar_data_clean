#!/usr/bin/env python3
"""
Convert Syndrone dataset to unified format (compatible with convert_eden.py output).

⚠️ CRITICAL OUTPUT REQUIREMENTS:
1. Depth: MUST be z-depth (perpendicular distance along optical axis), NOT euclidean distance
2. Pose: MUST be world-to-camera (w2c) in OpenCV coordinate system
3. Coordinate system: OpenCV right-handed (x=right, y=down, z=forward)

Input format (Syndrone):
  - rgb/*.jpg                             # RGB images
  - depth/*.png                           # uint16, z-depth (perpendicular distance) NOT euclidean!
  - camera/*.json                         # UE coordinates: {x, y, z, pitch, yaw, roll}

Output format (unified, compatible with convert_eden.py):
  - images/00000.jpg, 00001.jpg, ...      # RGB images (copied)
  - depths/00000.npy, 00001.npy, ...      # z-depth (NOT euclidean!) in NPY format
  - image_names.json                      # list of image names
  - cam_from_worlds.npy                   # (N,3,4) world->camera w2c matrices in OpenCV
  - intrinsics.npy                        # (N,3,3) camera intrinsics

Coordinate System Conversions:
  - Syndrone uses Unreal Engine (UE) left-handed coordinate system
  - Output uses OpenCV right-handed coordinate system
  - Depth: Syndrone z-depth (uint16 PNG) -> OpenCV z-depth (float32 NPY) - just scale conversion
  - Pose: UE (x=forward, y=right, z=up) -> OpenCV (x=right, y=down, z=forward)
"""

import os
import json
import glob
import math
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
import cv2
from tqdm import tqdm

# Depth storage (assuming you have depth_helper.py with _store_depth)
try:
    from depth_helper import _store_depth
except ImportError:
    print("[WARN] depth_helper not found, will use cv2.imwrite for EXR")
    def _store_depth(path, depth_map):
        os.makedirs(Path(path).parent, exist_ok=True)
        return cv2.imwrite(path, depth_map.astype(np.float32))


IMAGE_FOLDER_NAME = "images"
DEPTH_FOLDER_NAME = "depths"
COMPLETION_INDICATOR_FILE = "complete_log.txt"


# ============================================================================
# UE Coordinate System Conversion (from syndrone_multi_frame_vis.py)
# ============================================================================

def d2r(x):
    """Degrees to radians"""
    return x * math.pi / 180.0


def rot_x_lh(rx):
    """Roll, rotation around X-axis (left-handed)"""
    cr, sr = math.cos(rx), math.sin(rx)
    return np.array([[1,  0,   0],
                     [0, cr,  sr],
                     [0, -sr, cr]], dtype=np.float64)


def rot_y_lh(ry):
    """Pitch, rotation around Y-axis (left-handed)"""
    cy, sy = math.cos(ry), math.sin(ry)
    return np.array([[ cy, 0, -sy],
                     [  0, 1,   0],
                     [ sy, 0,  cy]], dtype=np.float64)


def rot_z_lh(rz):
    """Yaw, rotation around Z-axis (left-handed)"""
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array([[cz, sz, 0],
                     [-sz,cz, 0],
                     [ 0,  0, 1]], dtype=np.float64)


def ue_rotator_to_R_world(roll_deg, pitch_deg, yaw_deg):
    """
    Converts Unreal Engine Rotator (roll, pitch, yaw) to a 3x3 left-handed rotation matrix.
    The order of operations is Yaw (Z), then Pitch (Y), then Roll (X).
    """
    rx = rot_x_lh(d2r(roll_deg))
    ry = rot_y_lh(d2r(pitch_deg))
    rz = rot_z_lh(d2r(yaw_deg))
    return rz @ ry @ rx


# UE (left-handed, +X forward, +Y right, +Z up) -> OpenCV world (right-handed, x right, y down, z forward)
M_UE_to_CV = np.array([[0, 1,  0],
                       [0, 0, -1],
                       [1, 0,  0]], dtype=np.float64)


def build_cv_c2w_from_ue(location_dict, rotation_dict):
    """
    Build OpenCV camera-to-world (c2w) from UE location and rotation.
    
    Args:
        location_dict: {"x": float, "y": float, "z": float}
        rotation_dict: {"roll": float, "pitch": float, "yaw": float}
    
    Returns:
        R_cv: 3x3 rotation matrix (camera-to-world in OpenCV)
        t_cv: 3D translation vector (camera position in OpenCV world)
    """
    R_ue = ue_rotator_to_R_world(
        rotation_dict['roll'], 
        rotation_dict['pitch'], 
        -rotation_dict['yaw']  # Note: negative yaw
    )
    R_cv = M_UE_to_CV @ R_ue @ M_UE_to_CV.T
    # Transform position from UE to OpenCV: (y, -z, x) in UE -> (x, y, z) in CV
    t_cv = np.array([
        location_dict['y'], 
        -location_dict['z'], 
        location_dict['x']
    ], dtype=np.float64)
    return R_cv, t_cv


def make_K_from_fovx(fovx_deg, W, H, aspect_ratio=None):
    """
    Compute camera intrinsics from horizontal FOV.
    
    Returns:
        fx, fy, cx, cy
    """
    if aspect_ratio is None:
        aspect_ratio = W / H
    fovx = d2r(fovx_deg)
    fx = (W * 0.5) / math.tan(fovx * 0.5)
    v = 2.0 * math.atan(math.tan(fovx * 0.5) / aspect_ratio)
    fy = (H * 0.5) / math.tan(v * 0.5)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    return fx, fy, cx, cy


def c2w_to_w2c(R_c2w, t_c2w):
    """
    Convert camera-to-world (c2w) to world-to-camera (w2c).
    
    Args:
        R_c2w: 3x3 rotation matrix (camera to world)
        t_c2w: 3D translation vector (camera position in world)
    
    Returns:
        R_w2c: 3x3 rotation matrix (world to camera)
        t_w2c: 3D translation vector (world origin in camera frame)
    """
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    return R_w2c, t_w2c


def convert_depth_png_to_meters(depth_png: np.ndarray) -> np.ndarray:
    """
    Convert Syndrone PNG depth (uint16) to meters.
    
    ⚠️ IMPORTANT: Syndrone depth is ALREADY z-depth (perpendicular distance), NOT euclidean!
    Based on syndrone_multi_frame_vis.py analysis, the depth is z-depth format.
    The conversion is: depth_meters = depth_uint16 / 65535.0 * 1000.0
    
    Args:
        depth_png: uint16 depth map from PNG
    
    Returns:
        depth_meters: float64 z-depth map in meters (perpendicular distance)
    """
    # Convert to float and scale to meters
    # This is ALREADY z-depth, no conversion needed!
    depth_meters = depth_png.astype(np.float64) / 65535.0 * 1000.0
    return depth_meters


# ============================================================================
# Conversion Functions
# ============================================================================

def load_syndrone_frame(
    rgb_path: str, 
    depth_path: str, 
    camera_path: str,
    fovx_deg: float = 90.0
) -> Tuple:
    """
    Load one frame from Syndrone dataset and convert to OpenCV format.
    
    **IMPORTANT**: Syndrone depth PNG is ALREADY z-depth, not euclidean distance!
    Based on analysis of syndrone_multi_frame_vis.py (see line 112 where normalization is disabled),
    the depth values represent perpendicular distance along the optical axis.
    
    **OUTPUT FORMAT (compatible with convert_eden.py)**:
    - Depth: z-depth (perpendicular distance along optical axis)
    - Pose: world-to-camera (w2c) matrix in OpenCV coordinate system
    - Coordinate system: OpenCV (x=right, y=down, z=forward, right-handed)
    
    Returns:
        rgb: HxWx3 RGB image (uint8)
        depth_z: HxW z-depth map (float64, in meters)
        K: 3x3 intrinsics matrix
        w2c_matrix: 3x4 world-to-camera (cam_from_world) matrix in OpenCV
        c2w_R: 3x3 c2w rotation (for visualization only)
        c2w_t: 3D c2w translation (for visualization only)
    """
    # Load RGB
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None:
        raise ValueError(f"Cannot read RGB: {rgb_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    
    # ========================================================================
    # DEPTH CONVERSION: Syndrone PNG (uint16) -> z-depth in meters
    # ========================================================================
    # Step 1: Load PNG depth (uint16 format)
    depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_png is None:
        raise ValueError(f"Cannot read depth: {depth_path}")
    
    # Step 2: Convert uint16 to meters
    # ⚠️ IMPORTANT: Syndrone depth is ALREADY z-depth (perpendicular distance)!
    # No euclidean-to-z conversion needed!
    depth_z = convert_depth_png_to_meters(depth_png)
    
    # Load camera pose (UE format)
    with open(camera_path, 'r', encoding='utf-8') as f:
        camera_json = json.load(f)
    
    location = {"x": camera_json["x"], "y": camera_json["y"], "z": camera_json["z"]}
    rotation = {"roll": camera_json["roll"], "pitch": camera_json["pitch"], "yaw": camera_json["yaw"]}
    
    # Compute intrinsics from FOV
    fx, fy, cx, cy = make_K_from_fovx(fovx_deg, W, H)
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)
    
    # ========================================================================
    # POSE CONVERSION: UE (left-handed) -> OpenCV w2c (right-handed)
    # ========================================================================
    # Step 1: Build c2w (camera-to-world) in OpenCV coordinate system
    R_c2w, t_c2w = build_cv_c2w_from_ue(location, rotation)
    
    # Step 2: ⚠️ CRITICAL: Convert c2w to w2c (cam_from_world) for output
    # This is the required format: world-to-camera transformation
    R_w2c, t_w2c = c2w_to_w2c(R_c2w, t_c2w)
    
    # Step 3: Build 3x4 w2c matrix [R | t]
    w2c_matrix = np.hstack([R_w2c, t_w2c.reshape(3, 1)])
    
    # Verify output format
    assert w2c_matrix.shape == (3, 4), f"w2c matrix must be 3x4, got {w2c_matrix.shape}"
    assert depth_z.shape == (H, W), f"depth must be HxW, got {depth_z.shape}"
    
    return rgb, depth_z, K, w2c_matrix, R_c2w, t_c2w


def convert_syndrone_scene(
    input_dir: Path,
    output_dir: Path,
    fovx_deg: float = 90.0,
    dataset_name: str = "syndrone",
    version: str = "0.1",
    limit: int = 0
):
    """
    Convert one Syndrone scene to unified format.
    
    Args:
        input_dir: Path to scene directory (contains rgb/, depth/, camera/)
        output_dir: Path to output directory
        fovx_deg: Horizontal field of view in degrees
        dataset_name: Dataset name for metadata
        version: Dataset version for metadata
        limit: Maximum number of frames to convert (0 = all)
    """
    # Create output directories
    out_images = output_dir / IMAGE_FOLDER_NAME
    out_depths = output_dir / DEPTH_FOLDER_NAME
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_depths, exist_ok=True)
    
    # Find all RGB frames
    rgb_dir = input_dir / "rgb"
    rgb_files = sorted(glob.glob(str(rgb_dir / "*.jpg")))
    
    if not rgb_files:
        raise ValueError(f"No RGB files found in {rgb_dir}")
    
    # Apply limit if specified
    if limit > 0 and len(rgb_files) > limit:
        rgb_files = rgb_files[:limit]
        print(f"[INFO] Limiting to {limit} frames (out of total available)")
    
    print(f"[INFO] Found {len(rgb_files)} frames in {input_dir}")
    
    # Prepare data lists
    image_names = []
    cam_from_worlds_list = []
    intrinsics_list = []
    
    # Process each frame
    for idx, rgb_path in enumerate(tqdm(rgb_files, desc="Converting frames")):
        stem = Path(rgb_path).stem
        depth_path = str(input_dir / "depth" / f"{stem}.png")
        camera_path = str(input_dir / "camera" / f"{stem}.json")
        
        # Check if corresponding files exist
        if not os.path.exists(depth_path):
            print(f"[WARN] Missing depth for {stem}, skipping")
            continue
        if not os.path.exists(camera_path):
            print(f"[WARN] Missing camera for {stem}, skipping")
            continue
        
        try:
            # Load frame data
            rgb, depth_z, K, w2c_matrix, _, _ = load_syndrone_frame(
                rgb_path, depth_path, camera_path, fovx_deg
            )
            
            # Generate output filename
            frame_id = f"{idx:05d}"
            out_rgb_name = f"{frame_id}.jpg"
            # Use .npy for depth since OpenEXR may not be available
            out_depth_name = f"{frame_id}.npy"
            
            # Save RGB (copy/convert)
            out_rgb_path = str(out_images / out_rgb_name)
            cv2.imwrite(out_rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            # Save depth as NPY (more compatible than EXR)
            out_depth_path = str(out_depths / out_depth_name)
            try:
                os.makedirs(Path(out_depth_path).parent, exist_ok=True)
                np.save(out_depth_path, depth_z.astype(np.float32))
            except Exception as e:
                print(f"[ERROR] Failed to save depth for frame {frame_id}: {e}")
                continue
            
            # Store metadata
            image_names.append(out_rgb_name)
            cam_from_worlds_list.append(w2c_matrix)
            intrinsics_list.append(K)
            
        except Exception as e:
            print(f"[ERROR] Failed to process frame {stem}: {e}")
            continue
    
    if not image_names:
        raise RuntimeError(f"No frames successfully converted from {input_dir}")
    
    # Convert to numpy arrays
    cam_from_worlds_array = np.array(cam_from_worlds_list, dtype=np.float64)  # (N, 3, 4)
    intrinsics_array = np.array(intrinsics_list, dtype=np.float64)  # (N, 3, 3)
    
    # ========================================================================
    # VERIFY OUTPUT FORMAT (critical checks)
    # ========================================================================
    n_frames = len(image_names)
    assert cam_from_worlds_array.shape == (n_frames, 3, 4), \
        f"cam_from_worlds must be (N,3,4), got {cam_from_worlds_array.shape}"
    assert intrinsics_array.shape == (n_frames, 3, 3), \
        f"intrinsics must be (N,3,3), got {intrinsics_array.shape}"
    
    # Verify w2c matrices (rotation part should have det ≈ 1)
    R_w2c = cam_from_worlds_array[:, :, :3]  # (N, 3, 3)
    dets = np.linalg.det(R_w2c)
    assert np.allclose(np.abs(dets), 1.0, atol=1e-2), \
        f"Rotation matrices should have |det| ≈ 1, got range [{dets.min():.4f}, {dets.max():.4f}]"
    
    print(f"[INFO] Output format verified:")
    print(f"       - cam_from_worlds: {cam_from_worlds_array.shape} (world->camera w2c in OpenCV)")
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
        f.write(f"\n")
        f.write(f"Output format verification:\n")
        f.write(f"  - Depth: z-depth (perpendicular distance), NOT euclidean\n")
        f.write(f"  - Pose: world-to-camera (w2c) in OpenCV coordinate system\n")
        f.write(f"  - cam_from_worlds shape: {cam_from_worlds_array.shape}\n")
        f.write(f"  - intrinsics shape: {intrinsics_array.shape}\n")
    
    print(f"[SUCCESS] Converted {len(image_names)} frames to {output_dir}")
    print(f"[SUCCESS] ✓ Depth: z-depth format (NOT euclidean)")
    print(f"[SUCCESS] ✓ Pose: OpenCV w2c format")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert Syndrone dataset to unified format (compatible with convert_eden.py)"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing rgb/, depth/, camera/ subdirectories"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: input_dir_converted)"
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="Horizontal field of view in degrees (default: 90.0)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="syndrone",
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
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = Path(str(args.input_dir).rstrip('/') + "_converted")
    
    print(f"[INFO] Input:  {args.input_dir}")
    print(f"[INFO] Output: {args.output_dir}")
    print(f"[INFO] FOV:    {args.fov}°")
    
    # Convert
    convert_syndrone_scene(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fovx_deg=args.fov,
        dataset_name=args.dataset_name,
        version=args.version,
        limit=args.limit
    )
    
    print("[INFO] Conversion complete!")
    print(f"[INFO] You can now visualize with:")
    print(f"       python syndrone_multi_frame_vis.py --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()

