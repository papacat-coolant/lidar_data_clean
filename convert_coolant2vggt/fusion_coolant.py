#!/usr/bin/env python3
"""
Multi-view depth fusion for Coolant converted dataset.

Reads from the converted format:
  - images/00000.jpg, 00001.jpg, ...
  - depths/00000.npy, 00001.npy, ...
  - cam_from_worlds.npy (N, 3, 4) - world to camera transforms
  - intrinsics.npy (N, 3, 3)
  - image_names.json

Uses neighboring frames as source views for geometric consistency checking.
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def read_img(filename):
    """Read image and normalize to [0, 1]."""
    img = Image.open(filename)
    np_img = np.array(img, dtype=np.float32) / 255.0
    return np_img


def save_mask(filename, mask):
    """Save binary mask as uint8 image."""
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


def w2c_to_c2w(w2c):
    """Convert world-to-camera (3x4) to camera-to-world (4x4)."""
    R_w2c = w2c[:3, :3]
    t_w2c = w2c[:3, 3]
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c
    c2w = np.eye(4, dtype=w2c.dtype)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w
    return c2w


def c2w_to_w2c(c2w):
    """Convert camera-to-world (4x4) to world-to-camera (4x4)."""
    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    w2c = np.eye(4, dtype=c2w.dtype)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c
    return w2c


def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, 
                        depth_src, intrinsics_src, extrinsics_src):
    """Project reference depth to source view and back."""
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    
    # Step 1: project reference pixels to source view
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    
    # Reference 3D space (camera coordinates)
    xyz_ref = np.matmul(
        np.linalg.inv(intrinsics_ref),
        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1])
    )
    
    # Transform to source camera coordinates: src_from_ref = src_from_world @ world_from_ref
    # extrinsics_ref is world_from_ref (c2w), extrinsics_src is world_from_src (c2w)
    # We need: src_from_ref = inv(world_from_src) @ world_from_ref
    src_from_ref = np.linalg.inv(extrinsics_src) @ extrinsics_ref
    xyz_src = np.matmul(src_from_ref, np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    
    # Source view pixel coordinates
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / (K_xyz_src[2:3] + 1e-8)
    
    # Step 2: sample source depth at projected locations
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Handle NaN and invalid depths
    sampled_depth_src = np.nan_to_num(sampled_depth_src, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Step 3: project back to reference view
    # Only use valid sampled depths
    valid_mask = sampled_depth_src > 0.1
    sampled_depth_src_safe = sampled_depth_src.copy()
    sampled_depth_src_safe[~valid_mask.reshape(height, width)] = 1.0  # Avoid division by zero
    
    xyz_src_sampled = np.matmul(
        np.linalg.inv(intrinsics_src),
        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src_safe.reshape([-1])
    )
    
    ref_from_src = np.linalg.inv(extrinsics_ref) @ extrinsics_src
    xyz_reprojected = np.matmul(ref_from_src, np.vstack((xyz_src_sampled, np.ones_like(x_ref))))[:3]
    
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    
    # Mask out invalid reprojections
    depth_reprojected[~valid_mask.reshape(height, width)] = 0
    
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3] + 1e-8)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)
    
    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref,
                                depth_src, intrinsics_src, extrinsics_src,
                                abs_depth_threshold=0.1):
    """Check geometric consistency between reference and source depth."""
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref,
        depth_src, intrinsics_src, extrinsics_src
    )
    
    # Valid depth mask
    valid_depth = (depth_ref > 0.1) & (depth_reprojected > 0.1)
    
    # For aerial data with large baseline, pixel reprojection error is huge (hundreds of pixels)
    # So we ONLY use depth consistency check
    depth_diff = np.abs(depth_reprojected - depth_ref)
    
    # Use user-provided threshold (strict mode)
    mask = valid_depth & (depth_diff < abs_depth_threshold)
    
    print(f"  Depth consistency: threshold={abs_depth_threshold:.2f}m, inliers={mask.sum()}/{valid_depth.sum()} ({100.0*mask.sum()/max(valid_depth.sum(),1):.1f}%)")
    
    depth_reprojected[~mask] = 0
    return mask, depth_reprojected


def process_view(args_tuple):
    """Process one reference view with its source views."""
    (ref_idx, src_indices, data_dir, num_src_views, depth_threshold) = args_tuple
    
    # Load paths
    images_dir = data_dir / "images"
    depths_dir = data_dir / "depths"
    
    with open(data_dir / "image_names.json", "r") as f:
        image_names = json.load(f)
    
    cam_from_worlds = np.load(data_dir / "cam_from_worlds.npy")  # (N, 3, 4) w2c
    intrinsics_array = np.load(data_dir / "intrinsics.npy")  # (N, 3, 3)
    
    # Reference view data
    ref_img_path = images_dir / image_names[ref_idx]
    ref_depth_path = depths_dir / Path(image_names[ref_idx]).with_suffix(".npy")
    
    ref_img = read_img(str(ref_img_path))
    ref_depth = np.load(str(ref_depth_path)).astype(np.float32)
    
    ref_intrinsics = intrinsics_array[ref_idx]
    ref_w2c = cam_from_worlds[ref_idx]  # (3, 4)
    ref_extrinsics = w2c_to_c2w(ref_w2c)  # Convert to c2w (4x4)
    
    height, width = ref_depth.shape
    
    # Simple photometric mask (non-zero depth)
    photo_mask = ref_depth > 0.1
    
    # Geometric consistency check with source views
    all_depth_reprojected = []
    geo_mask_sum = np.zeros((height, width), dtype=np.int32)
    
    for src_idx in src_indices:
        src_depth_path = depths_dir / Path(image_names[src_idx]).with_suffix(".npy")
        src_depth = np.load(str(src_depth_path)).astype(np.float32)
        
        src_intrinsics = intrinsics_array[src_idx]
        src_w2c = cam_from_worlds[src_idx]
        src_extrinsics = w2c_to_c2w(src_w2c)
        
        geo_mask, depth_reprojected = check_geometric_consistency(
            ref_depth, ref_intrinsics, ref_extrinsics,
            src_depth, src_intrinsics, src_extrinsics,
            abs_depth_threshold=depth_threshold
        )
        
        geo_mask_sum += geo_mask.astype(np.int32)
        all_depth_reprojected.append(depth_reprojected)
    
    # Require consistency with at least half of source views
    geo_mask = geo_mask_sum >= (len(src_indices) // 2)
    
    # Average depth from all consistent views
    depth_sum = sum(all_depth_reprojected) + ref_depth
    depth_est_averaged = depth_sum / (geo_mask_sum + 1)
    
    # Filter outliers
    if depth_est_averaged[geo_mask].size > 0:
        depth_95 = np.quantile(depth_est_averaged[geo_mask], 0.95)
        geo_mask = geo_mask & (depth_est_averaged < depth_95)
    
    # Erosion to remove boundary noise
    geo_mask = cv2.erode(geo_mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1)
    
    final_mask = np.logical_and(photo_mask, geo_mask)
    
    # Save outputs in input directory structure
    mask_dir = data_dir / "mask"
    fused_depth_dir = data_dir / "fused_depths"
    
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(fused_depth_dir, exist_ok=True)
    
    frame_name = Path(image_names[ref_idx]).stem
    
    # Save fused depth
    np.save(fused_depth_dir / f"{frame_name}.npy", depth_est_averaged.astype(np.float32))
    
    # Visualize depth
    depth_vis = depth_est_averaged * final_mask.astype(np.float32)
    depth_vis = np.log(depth_vis + 1)
    depth_normalized = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
    depth_color = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(fused_depth_dir / f"{frame_name}.png"), depth_color)
    
    # Save masks
    save_mask(str(mask_dir / f"{frame_name}_photo.png"), photo_mask)
    save_mask(str(mask_dir / f"{frame_name}_geo.png"), geo_mask)
    save_mask(str(mask_dir / f"{frame_name}_final.png"), final_mask)
    
    print(f"Frame {ref_idx:04d}: photo/geo/final = {photo_mask.mean():.3f}/{geo_mask.mean():.3f}/{final_mask.mean():.3f}")
    
    # Generate point cloud
    valid_points = final_mask
    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
    color = ref_img[valid_points]
    
    xyz_ref = np.matmul(
        np.linalg.inv(ref_intrinsics),
        np.vstack((x, y, np.ones_like(x))) * depth
    )
    xyz_world = np.matmul(ref_extrinsics, np.vstack((xyz_ref, np.ones_like(x))))[:3]
    
    return (xyz_world.T, (color * 255).astype(np.uint8))


def fusion_coolant(data_dir: Path, num_src_views: int = 5, num_workers: int = None, 
                   depth_threshold: float = 0.1):
    """
    Fuse multi-view depths from converted Coolant dataset.
    
    Output structure (in data_dir):
        mask/{frame}_photo.png, {frame}_geo.png, {frame}_final.png
        fused_depths/{frame}.npy, {frame}.png
        fused.ply
    
    Args:
        data_dir: Path to converted dataset (contains cam_from_worlds.npy, etc.)
        num_src_views: Number of neighboring frames to use as source views
        num_workers: Number of parallel workers (default: CPU count)
    """
    data_dir = Path(data_dir)
    
    with open(data_dir / "image_names.json", "r") as f:
        image_names = json.load(f)
    
    cam_from_worlds = np.load(data_dir / "cam_from_worlds.npy")  # Need this for frame skipping check
    num_frames = len(image_names)
    
    # Build view pairs (reference + neighboring source views)
    # Skip first frame if it has large rotation difference (likely calibration frame)
    start_frame = 0
    if num_frames > 2:
        R0 = cam_from_worlds[0][:3, :3]
        R1 = cam_from_worlds[1][:3, :3]
        R_diff = R0.T @ R1
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        if np.degrees(angle) > 30:
            start_frame = 1
            print(f"[INFO] Skipping frame 0 (rotation change: {np.degrees(angle):.1f}°)")
    
    view_pairs = []
    for ref_idx in range(start_frame, num_frames):
        # Select neighboring frames as source views
        src_indices = []
        for offset in range(1, num_src_views + 1):
            if ref_idx - offset >= start_frame:
                src_indices.append(ref_idx - offset)
            if ref_idx + offset < num_frames:
                src_indices.append(ref_idx + offset)
        
        if len(src_indices) >= 2:  # Need at least 2 source views
            view_pairs.append((ref_idx, src_indices, data_dir, num_src_views, depth_threshold))
    
    print(f"Processing {len(view_pairs)} frames with {num_src_views} neighboring source views each")
    
    # Process in parallel (or single-threaded for debugging)
    if num_workers is None:
        num_workers = os.cpu_count()
    
    if num_workers == 1:
        # Single-threaded mode - allows ipdb breakpoints
        results = []
        for view_pair in tqdm(view_pairs, desc="Fusing depths"):
            results.append(process_view(view_pair))
    else:
        # Multi-threaded mode
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_view, view_pairs), total=len(view_pairs), desc="Fusing depths"))
    
    # Merge point clouds
    print("Merging point clouds...")
    all_vertices = []
    all_colors = []
    
    for xyz_world, color in results:
        if xyz_world.shape[0] > 0:
            all_vertices.append(xyz_world)
            all_colors.append(color)
    
    vertices = np.concatenate(all_vertices, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    
    # Subsample for visualization
    subsample = 4
    vertices = vertices[::subsample]
    colors = colors[::subsample]
    
    print(f"Total points: {len(vertices):,}")
    
    # Save as PLY in data_dir
    try:
        from plyfile import PlyData, PlyElement
        vertex_data = np.array(
            [(v[0], v[1], v[2], c[0], c[1], c[2]) for v, c in zip(vertices, colors)],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )
        ply_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([ply_element]).write(str(data_dir / 'fused.ply'))
        print(f"Point cloud saved to {data_dir / 'fused.ply'}")
    except Exception as e:
        print(f"[WARN] Failed to save PLY: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fuse multi-view depths for Coolant dataset")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to converted Coolant dataset (outputs will be saved in this directory)"
    )
    parser.add_argument(
        "--num_src_views",
        type=int,
        default=5,
        help="Number of neighboring frames to use as source views (default: 5)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count, use 1 for debugging)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (single-threaded, allows ipdb breakpoints)"
    )
    parser.add_argument(
        "--depth_threshold",
        type=float,
        default=0.1,
        help="Absolute depth error threshold in meters (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Force single-threaded for debugging
    if args.debug:
        args.num_workers = 1
        print("[DEBUG] Debug mode enabled - running single-threaded")
    
    print(f"[INFO] Data directory: {args.data_dir}")
    print(f"[INFO] Outputs will be saved in:")
    print(f"       - {args.data_dir}/mask/")
    print(f"       - {args.data_dir}/fused_depths/")
    print(f"       - {args.data_dir}/fused.ply")
    print(f"[INFO] Source views per frame: {args.num_src_views}")
    print(f"[INFO] Depth error threshold: {args.depth_threshold}m")
    print(f"[INFO] Workers: {args.num_workers if args.num_workers else 'auto'}")
    
    fusion_coolant(
        data_dir=args.data_dir,
        num_src_views=args.num_src_views,
        num_workers=args.num_workers,
        depth_threshold=args.depth_threshold
    )
    
    print("[INFO] Fusion complete!")


if __name__ == "__main__":
    main()

