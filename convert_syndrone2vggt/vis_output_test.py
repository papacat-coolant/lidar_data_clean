#!/usr/bin/env python3
"""
Visualize output_test_100 format data with nerfvis
- Depth: z-depth (perpendicular distance), NOT euclidean
- Pose: world-to-camera (w2c) in OpenCV coordinate system
- cam_from_worlds shape: (N, 3, 4)
- intrinsics shape: (N, 3, 3)
"""

import os
import json
import argparse
import numpy as np
import cv2
import nerfvis


def backproject_z_depth(depth, fx, fy, cx, cy, stride=2):
    """
    Backproject z-depth to 3D camera coordinates.
    For z-depth: z = depth[u,v], then x = z*(u-cx)/fx, y = z*(v-cy)/fy
    
    Args:
        depth: (H, W) array, z-depth values
        fx, fy, cx, cy: camera intrinsics
        stride: downsampling stride
    
    Returns:
        pts_cam: (N, 3) points in camera coordinates
        uu, vv: (N,) pixel coordinates
    """
    H, W = depth.shape[:2]
    us = np.arange(0, W, stride)
    vs = np.arange(0, H, stride)
    uu, vv = np.meshgrid(us, vs)
    
    # Get depth values at sampled pixels
    z = depth[::stride, ::stride]  # (H/stride, W/stride)
    
    # Compute x, y from z-depth
    x = z * (uu - cx) / fx
    y = z * (vv - cy) / fy
    
    # Stack to get 3D points
    pts_cam = np.stack([x, y, z], axis=-1)  # (H/stride, W/stride, 3)
    pts_cam = pts_cam.reshape(-1, 3)
    
    return pts_cam, uu.reshape(-1), vv.reshape(-1)


def w2c_to_c2w(w2c):
    """
    Convert world-to-camera (3x4) to camera-to-world
    
    Args:
        w2c: (3, 4) [R|t] matrix where P_cam = R @ P_world + t
    
    Returns:
        R_c2w: (3, 3) rotation matrix
        t_c2w: (3,) translation vector
    """
    R_w2c = w2c[:3, :3]  # (3, 3)
    t_w2c = w2c[:3, 3]   # (3,)
    
    # c2w = w2c^-1
    # For [R|t] matrix: inv = [R^T | -R^T @ t]
    R_c2w = R_w2c.T
    t_c2w = -R_w2c.T @ t_w2c
    
    return R_c2w, t_c2w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Path to output directory containing cam_from_worlds.npy, intrinsics.npy, images/, depths/")
    ap.add_argument("--limit", type=int, default=0, help="Maximum number of frames to visualize (0=all)")
    ap.add_argument("--stride", type=int, default=4, help="Point cloud downsampling stride")
    ap.add_argument("--point_size", type=float, default=1.0)
    ap.add_argument("--z_size", type=float, default=0.3, help="Camera frustum depth")
    ap.add_argument("--keep_pct", type=float, default=95.0, 
                    help="Keep near distance percentile (0-100), e.g. 95 means discard farthest 5% noise points")
    ap.add_argument("--min_depth", type=float, default=0.1, help="Minimum depth clipping (meters)")
    ap.add_argument("--max_depth", type=float, default=0.0, help="Maximum depth clipping (meters, 0=no clipping)")
    ap.add_argument("--center", action="store_true", help="Center the point cloud at origin")
    args = ap.parse_args()
    
    # Load data
    data_dir = args.data_dir
    cam_from_worlds = np.load(os.path.join(data_dir, "cam_from_worlds.npy"))  # (N, 3, 4)
    intrinsics = np.load(os.path.join(data_dir, "intrinsics.npy"))  # (N, 3, 3)
    
    with open(os.path.join(data_dir, "image_names.json"), "r") as f:
        image_names = json.load(f)
    
    N = len(image_names)
    if args.limit > 0:
        N = min(N, args.limit)
    
    print(f"[INFO] Loading {N} frames from {data_dir}")
    print(f"[INFO] cam_from_worlds shape: {cam_from_worlds.shape}")
    print(f"[INFO] intrinsics shape: {intrinsics.shape}")
    print(f"[INFO] Depth type: z-depth (perpendicular distance)")
    print(f"[INFO] Pose type: world-to-camera (w2c)")
    
    # Create scene
    scene = nerfvis.Scene("Output Test 100 Visualization", default_opencv=True)
    scene.set_opencv()
    scene.set_opencv_world()
    
    # First pass: collect all data
    frame_data = []
    all_points_for_center = []
    
    # Process each frame
    for i in range(N):
        img_name = image_names[i]
        stem = os.path.splitext(img_name)[0]
        
        # Load image
        img_path = os.path.join(data_dir, "images", img_name)
        rgb_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        assert rgb_bgr is not None, f"Cannot read image: {img_path}"
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        
        # Load depth
        depth_path = os.path.join(data_dir, "depths", f"{stem}.npy")
        depth = np.load(depth_path).astype(np.float64)
        
        # Get camera parameters
        w2c = cam_from_worlds[i]  # (3, 4)
        K = intrinsics[i]  # (3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Convert w2c to c2w
        R_c2w, t_c2w = w2c_to_c2w(w2c)
        
        # Backproject depth to 3D points in camera coordinates
        pts_cam, uu, vv = backproject_z_depth(depth, fx, fy, cx, cy, stride=args.stride)
        
        # Filter depth
        d_sub = depth[::args.stride, ::args.stride]
        keep = np.ones_like(d_sub, dtype=bool)
        
        if args.keep_pct > 0 and args.keep_pct < 100:
            thr = np.percentile(d_sub[d_sub > 0], args.keep_pct)
            keep &= (d_sub <= thr)
        
        if args.min_depth > 0:
            keep &= (d_sub >= args.min_depth)
        
        if args.max_depth > 0:
            keep &= (d_sub <= args.max_depth)
        
        keep = keep.reshape(-1)
        
        # Apply filter
        pts_cam = pts_cam[keep]
        uu_keep, vv_keep = uu[keep], vv[keep]
        colors = (rgb[vv_keep, uu_keep, :] / 255.0).reshape(-1, 3)
        
        # Transform to world coordinates
        pts_world = (R_c2w @ pts_cam.T).T + t_c2w[None, :]
        
        # Store frame data
        group = f"{i:04d}_{stem}"
        frame_data.append({
            'group': group,
            'R_c2w': R_c2w,
            't_c2w': t_c2w,
            'fx': fx,
            'W': W,
            'H': H,
            'rgb': rgb,
            'pts_world': pts_world,
            'colors': colors
        })
        
        # Collect points for centering calculation
        all_points_for_center.append(pts_world)
        
        if (i + 1) % 10 == 0:
            print(f"[INFO] Processed {i + 1}/{N} frames")
    
    # Calculate center offset
    center_offset = np.zeros(3)
    if args.center and len(all_points_for_center) > 0:
        all_pts = np.concatenate(all_points_for_center, axis=0)
        center_offset = np.mean(all_pts, axis=0)
        print(f"[INFO] Point cloud center: {center_offset}")
        print(f"[INFO] Centering at origin...")
    
    # Second pass: add to scene with centering
    for data in frame_data:
        # Apply centering
        t_centered = data['t_c2w'] - center_offset
        pts_centered = data['pts_world'] - center_offset[None, :]
        
        # Camera frustum
        scene.add_camera_frustum(
            f"camera/{data['group']}/frustum",
            r=data['R_c2w'], t=t_centered,
            focal_length=float(data['fx']),
            image_width=data['W'], image_height=data['H'],
            z=float(args.z_size)
        )
        
        # Camera image
        scene.add_image(
            f"camera/{data['group']}/image",
            data['rgb'], r=data['R_c2w'], t=t_centered,
            focal_length=float(data['fx']),
            z=float(args.z_size), image_size=min(1024, max(data['W'], data['H']))
        )
        
        # Point cloud
        scene.add_points(
            f"points/{data['group']}",
            pts_centered, point_size=args.point_size, vert_color=data['colors']
        )
    
    print(f"[INFO] Visualization complete. Total frames: {N}")
    scene.add_axes()
    scene.display()


if __name__ == "__main__":
    main()

