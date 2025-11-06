# vis_nerfvis_multi.py
# Multi-frame: camera pyramids + textured images + colored point clouds (depth as "euclidean distance from point to camera center")
# Optional: --global_extrinsic transforms all cameras and point clouds to "global world coordinate system"
#          Supports .npy/.npz (matrix name T or 4x4 array), .json ({"matrix":[[...],[...],...]} or {"R":..., "t":...})

import os, re, glob, json, math, argparse
import numpy as np
import cv2
import nerfvis

# def d2r(x): return x * math.pi / 180.0

# def rot_x(rx):
#     cr, sr = math.cos(rx), math.sin(rx)
#     return np.array([[1, 0, 0],
#                      [0, cr,-sr],
#                      [0, sr, cr]], dtype=np.float64)

# def rot_y(ry):
#     cy, sy = math.cos(ry), math.sin(ry)
#     return np.array([[ cy, 0, sy],
#                      [  0, 1,  0],
#                      [-sy, 0, cy]], dtype=np.float64)

# def rot_z(rz):
#     cz, sz = math.cos(rz), math.sin(rz)
#     return np.array([[cz,-sz, 0],
#                      [sz, cz, 0],
#                      [ 0,  0, 1]], dtype=np.float64)

# def ue_rotator_to_R_world(roll_deg, pitch_deg, yaw_deg):
#     rx = rot_x(d2r(roll_deg))
#     ry = rot_y(d2r(pitch_deg))
#     rz = rot_z(d2r(yaw_deg))
#     return rz @ ry @ rx  # C2W(UE)

def d2r(x):
    return x * math.pi / 180.0

def rot_x_lh(rx):
    """Roll, rotation around X-axis"""
    cr, sr = math.cos(rx), math.sin(rx)
    return np.array([[1,  0,   0],
                     [0, cr,  sr],
                     [0, -sr, cr]], dtype=np.float64)
def rot_y_lh(ry):
    """Pitch, rotation around Y-axis"""
    cy, sy = math.cos(ry), math.sin(ry)
    return np.array([[ cy, 0, -sy],
                     [  0, 1,   0],
                     [ sy, 0,  cy]], dtype=np.float64)
def rot_z_lh(rz):
    """Yaw, rotation around Z-axis"""
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
    # The multiplication order is R_z * R_y * R_x
    # This corresponds to an extrinsic ZYX rotation, which matches UE's convention.
    return rz @ ry @ rx

def make_K_from_fovx(fovx_deg, W, H, aspect_ratio=None):
    if aspect_ratio is None:
        aspect_ratio = W / H
    fovx = d2r(fovx_deg)
    fx = (W * 0.5) / math.tan(fovx * 0.5)
    v = 2.0 * math.atan(math.tan(fovx * 0.5) / aspect_ratio)
    fy = (H * 0.5) / math.tan(v * 0.5)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    return fx, fy, cx, cy

# UE (left-handed, +X forward, +Y right, +Z up) -> OpenCV world (right-handed, x right, y down, z forward) "axis mapping"
M_UE_to_CV = np.array([[0, 1,  0],
                       [0, 0, -1],
                       [1, 0,  0]], dtype=np.float64)

def build_cv_c2w_from_ue(location, rotation):
    R_ue = ue_rotator_to_R_world(rotation['roll'], rotation['pitch'], -rotation['yaw'])
    R_cv = M_UE_to_CV @ R_ue @ M_UE_to_CV.T
    t_cv = np.array([location['y'], -location['z'], location['x']], dtype=np.float64) 
    return R_cv, t_cv

def to_hom(X):  # (N,3)->(N,4)
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])

def backproject_points_from_euclidean_depth(depth, fx, fy, cx, cy, stride=2):
    """
    depth[u,v] is the "euclidean distance from point to camera center" (distance along ray direction).
    For each pixel (u,v):
      First construct ray direction d_cam = [x, y, 1], where x=(u-cx)/fx, y=(v-cy)/fy
      Normalize u_cam = d_cam / ||d_cam||
      Then 3D point (camera coordinate system) = depth * u_cam
    """
    H, W = depth.shape[:2]
    us = np.arange(0, W, stride)
    vs = np.arange(0, H, stride)
    uu, vv = np.meshgrid(us, vs)
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    ones = np.ones_like(x)
    dirs = np.stack([x, y, ones], axis=-1)  # (H/str, W/str, 3)
    # norms = np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8
    unit_dirs = dirs # / norms
    d = depth[::stride, ::stride][..., None]  # (.,.,1)
    pts_cam = unit_dirs * d                  # (.,.,3)
    pts_cam = pts_cam.reshape(-1, 3)
    return pts_cam, uu.reshape(-1), vv.reshape(-1)

def robust_load_depth(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".png":
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # PNG stores uint16 depth in centimeters*? Assume 1 unit = 0.01m (centimeter)
        arr = img.astype(np.float64) / 65535.0 * 1000.0
    else:
        raise ValueError(f"Unsupported depth file extension: {path}")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
    return arr

def load_global_extrinsic(path):
    if path is None:
        T = np.eye(4, dtype=np.float64)
        return T, T[:3,:3], T[:3,3]
    ext = os.path.splitext(path)[1].lower()
    if ext in [".npy", ".npz"]:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "T" in data:
                T = np.asarray(data["T"], dtype=np.float64)
            elif "matrix" in data:
                T = np.asarray(data["matrix"], dtype=np.float64)
            elif "R" in data and "t" in data:
                R = np.asarray(data["R"], dtype=np.float64); t = np.asarray(data["t"], dtype=np.float64).reshape(3)
                T = np.eye(4, dtype=np.float64); T[:3,:3]=R; T[:3,3]=t
            else:
                # 取第一个数组
                key = list(data.files)[0]
                T = np.asarray(data[key], dtype=np.float64)
        else:
            T = np.asarray(data, dtype=np.float64)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            js = json.load(f)
        if "matrix" in js:
            T = np.asarray(js["matrix"], dtype=np.float64)
        elif "R" in js and "t" in js:
            R = np.asarray(js["R"], dtype=np.float64); t = np.asarray(js["t"], dtype=np.float64).reshape(3)
            T = np.eye(4, dtype=np.float64); T[:3,:3]=R; T[:3,3]=t
        else:
            T = np.asarray(js, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported extrinsic file type: {path}")
    assert T.shape == (4,4), f"Extrinsic matrix must be 4x4, got {T.shape}"
    return T, T[:3,:3], T[:3,3]

# ---------- Pairing ----------
TS_RE = re.compile(r"_(\d{8}_\d{6})\.")

def pair_by_timestamp(output_dir, prefix):
    rgbs  = glob.glob(os.path.join(output_dir, f"{prefix}_rgb_*.png"))
    depths= glob.glob(os.path.join(output_dir, f"{prefix}_depth_*.npy"))
    poses = glob.glob(os.path.join(output_dir, f"{prefix}_pose_*.json"))
    def ts_map(paths):
        m = {}
        for p in paths:
            mobj = TS_RE.search(p)
            if mobj: m[mobj.group(1)] = p
        return m
    mr, md, mp = ts_map(rgbs), ts_map(depths), ts_map(poses)
    ts = sorted(set(mr.keys()) & set(md.keys()) & set(mp.keys()))
    pairs = [(mr[t], md[t], mp[t], t) for t in ts]
    return pairs

# ---------- Main Process ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="output", help="Directory containing RGB/Depth/Pose files")
    ap.add_argument("--prefix", default="render", help="File prefix, e.g. render_*")
    ap.add_argument("--limit", type=int, default=0, help="Maximum number of frames to visualize (0=all)")
    ap.add_argument("--stride", type=int, default=4, help="Point cloud downsampling stride")
    ap.add_argument("--point_size", type=float, default=1.0)
    ap.add_argument("--z_size", type=float, default=0.3, help="Camera frustum depth")
    ap.add_argument("--keep_pct", type=float, default=80.0, help="Keep near distance percentile (0-100), e.g. 95 means discard farthest 5% noise points")
    ap.add_argument("--min_depth", type=float, default=0.0, help="Minimum depth clipping (meters)")
    ap.add_argument("--max_depth", type=float, default=0.0, help="Maximum depth clipping (meters, 0=no clipping)")
    ap.add_argument("--global_extrinsic", type=str, help="Global extrinsic 4x4 (.npy/.npz/.json), maps CV world to global world")
    ap.add_argument("--center_global", action="store_true", help="After merging, perform global mean centering on all points")
    # --- Syndrone sequential dataset (rgb/depth/camera) ---
    ap.add_argument("--data_dir", type=str,
                    help="Path to dataset directory that contains rgb/, depth/, camera/ sub-folders (e.g. .../height20m)")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Build frame list depending on input mode
    # ------------------------------------------------------------------
    if args.data_dir:
        def make_seq_pairs(base_dir):
            rgbs = sorted(glob.glob(os.path.join(base_dir, "rgb", "*.jpg")))
            pairs = []
            for rgb_path in rgbs:
                stem = os.path.splitext(os.path.basename(rgb_path))[0]
                depth_path = os.path.join(base_dir, "depth", f"{stem}.png")
                cam_path   = os.path.join(base_dir, "camera", f"{stem}.json")
                if os.path.exists(depth_path) and os.path.exists(cam_path):
                    pairs.append((rgb_path, depth_path, cam_path, stem))
            return pairs
        pairs = make_seq_pairs(args.data_dir)
        if args.limit > 0:
            pairs = pairs[:args.limit]
        assert pairs, f"No frames found under {args.data_dir} (expecting rgb/depth/camera sub-folders)"
    else:
        pairs = pair_by_timestamp(args.output_dir, args.prefix)
        if args.limit > 0:
            pairs = pairs[:args.limit]
        assert pairs, f"Cannot find {args.prefix}_rgb/depth/pose_* files in {args.output_dir}"

    T_g, R_g, t_g = load_global_extrinsic(args.global_extrinsic) if args.global_extrinsic else (np.eye(4), np.eye(3), np.zeros(3))
    print(f"[INFO] Global extrinsic:\n{T_g}")

    scene = nerfvis.Scene("UE multi-capture (RGB + Depth + Cameras)", default_opencv=True)
    scene.set_opencv()
    scene.set_opencv_world()

    all_pts = []
    # Frame by frame
    pre_pose = None
    for i, (rgb_path, depth_path, pose_path, ts) in enumerate(pairs):
        # Read data
        # if i < 3:
        #     pre_pose = pose_path
        #     continue
        
        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        assert rgb_bgr is not None, f"Cannot read RGB: {rgb_path}"
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        depth = robust_load_depth(depth_path)

        with open(pose_path, "r", encoding="utf-8") as f:
            pose = json.load(f)
        # Support both original pose format and Syndrone simple format
        if "location" in pose:
            location = pose["location"]
            rotation = pose["rotation"]
            fovx_deg = float(pose.get("fov", 90.0))
            aspect_ratio = float(pose.get("aspect_ratio", W / H))
        else:
            location = {"x": pose["x"], "y": pose["y"], "z": pose["z"]}
            rotation = {"roll": pose["roll"], "pitch": pose["pitch"], "yaw": pose["yaw"]}
            fovx_deg = 90.0
            aspect_ratio = W / H

        fx, fy, cx, cy = make_K_from_fovx(fovx_deg, W, H, aspect_ratio)
        R_cv, t_cv = build_cv_c2w_from_ue(location, rotation)
        pts_cam, uu, vv = backproject_points_from_euclidean_depth(depth, fx, fy, cx, cy, stride=args.stride)
        d_sub = depth[::args.stride, ::args.stride]
        keep = np.ones_like(d_sub, dtype=bool)
        if args.keep_pct > 0 and args.keep_pct < 100:
            thr = np.percentile(d_sub, args.keep_pct)
            keep &= (d_sub <= thr)
        if args.min_depth > 0:
            keep &= (d_sub >= args.min_depth)
        if args.max_depth > 0:
            keep &= (d_sub <= args.max_depth)
        keep = keep.reshape(-1)

        pts_cam = pts_cam[keep]
        uu_keep, vv_keep = uu[keep], vv[keep]
        colors = (rgb[vv_keep, uu_keep, :] / 255.0).reshape(-1, 3)
        pts_world = (R_cv @ pts_cam.T).T + t_cv[None, :]
        pts_world = pts_world
        all_pts.append(pts_world)
        group = f"{i:04d}_{ts}"
        scene.add_camera_frustum(
            f"camera/{group}/frustum",
            r=R_cv, t=t_cv,
            focal_length=float(fx),
            image_width=W, image_height=H,
            z=float(args.z_size)
        )
        scene.add_image(
            f"camera/{group}/image",
            rgb, r=R_cv, t=t_cv,
            focal_length=float(fx),
            z=float(args.z_size), image_size=min(1024, max(W, H))
        )
        scene.add_points(
            f"points/{group}",
            pts_world, point_size=args.point_size, vert_color=colors
        )
        pre_pose = pose_path

    scene.add_axes()
    scene.display()

if __name__ == "__main__":
    main()
