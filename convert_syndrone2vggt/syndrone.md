# Syndrone Dataset Tools

## Overview

This directory contains tools for converting and visualizing the Syndrone dataset in a unified format.

**Key Features:**
- Converts Syndrone data (Unreal Engine format) to OpenCV coordinate system
- Handles z-depth format (perpendicular distance, NOT euclidean)
- Generates world-to-camera (w2c) pose matrices
- Visualizes 3D point clouds with camera frustums using nerfvis

## Download Dataset

Download the Syndrone dataset:
```bash
wget https://lttm.dei.unipd.it/paper_data/syndrone/syndrone.zip
unzip syndrone.zip
```

Expected structure:
```
Town01_Opt_120/Town01_Opt_120/ClearNoon/
├── height20m/
│   ├── rgb/          # RGB images (*.jpg)
│   ├── depth/        # Depth maps (*.png, uint16)
│   └── camera/       # Camera poses (*.json, UE format)
├── height50m/
└── height80m/
```

---

## Tool 1: `convert_syndrone.py`

Converts a single Syndrone scene to unified format.

### Usage

```bash
python convert_syndrone.py <INPUT_DIR> [OPTIONS]
```

### Arguments

- `INPUT_DIR`: Path to scene directory (must contain `rgb/`, `depth/`, `camera/` subdirectories)
- `--output_dir PATH`: Output directory (default: `{INPUT_DIR}_converted`)
- `--fov FLOAT`: Horizontal field of view in degrees (default: 90.0)
- `--dataset_name STR`: Dataset name for metadata (default: "syndrone")
- `--version STR`: Dataset version (default: "0.1")
- `--limit INT`: Max frames to convert, 0=all (default: 0)

### Examples

**Convert single scene:**
```bash
python convert_syndrone.py Town01_Opt_120/Town01_Opt_120/ClearNoon/height20m
```

**Convert with custom output directory:**
```bash
python convert_syndrone.py Town01_Opt_120/Town01_Opt_120/ClearNoon/height20m \
    --output_dir output_height20m \
    --fov 90.0
```

**Convert first 100 frames only:**
```bash
python convert_syndrone.py Town01_Opt_120/Town01_Opt_120/ClearNoon/height50m \
    --output_dir output_test_100 \
    --limit 100
```

### Output Format

The conversion creates the following structure:
```
output_dir/
├── images/
│   ├── 00000.jpg
│   ├── 00001.jpg
│   └── ...
├── depths/
│   ├── 00000.npy          # z-depth in meters (float32)
│   ├── 00001.npy
│   └── ...
├── image_names.json        # List of image filenames
├── cam_from_worlds.npy     # (N, 3, 4) w2c matrices in OpenCV
├── intrinsics.npy          # (N, 3, 3) camera intrinsics
└── complete_log.txt        # Conversion metadata
```

**Important Notes:**
- **Depth format**: z-depth (perpendicular distance along optical axis), NOT euclidean distance
- **Pose format**: world-to-camera (w2c) transformation matrices
- **Coordinate system**: OpenCV right-handed (x=right, y=down, z=forward)

---

## Tool 2: `batch_convert.sh`

Batch converts multiple Syndrone scenes at once.

### Configuration

Edit the script to set your paths:
```bash
INPUT_BASE="Town01_Opt_120/Town01_Opt_120/ClearNoon"
OUTPUT_BASE="output_unified"
FOV=90.0

SCENES=(
    "height20m"
    "height50m"
    "height80m"
)
```

### Usage

```bash
chmod +x batch_convert.sh
./batch_convert.sh
```

### Features

- **Skip existing**: Automatically skips scenes that have `complete_log.txt`
- **Error handling**: Continues on error, reports failures
- **Progress feedback**: Color-coded output for each scene

### Example Output

```
=========================================
Syndrone Batch Conversion Script
=========================================

Processing: height20m
  [INFO] Converting Town01_Opt_120/.../height20m -> output_unified/height20m
  [SUCCESS] Converted height20m

Processing: height50m
  [INFO] Already converted, skipping...

Processing: height80m
  [INFO] Converting Town01_Opt_120/.../height80m -> output_unified/height80m
  [SUCCESS] Converted height80m
```

---

## Tool 3: `vis_output_test.py`

Visualizes converted data using nerfvis with 3D point clouds and camera frustums.

### Prerequisites

```bash
pip install nerfvis opencv-python numpy
```

### Usage

```bash
python vis_output_test.py --data_dir <OUTPUT_DIR> [OPTIONS]
```

### Arguments

- `--data_dir PATH`: **Required**. Path to converted output directory
- `--limit INT`: Max frames to visualize, 0=all (default: 0)
- `--stride INT`: Point cloud downsampling stride (default: 4)
- `--point_size FLOAT`: Point size in visualization (default: 1.0)
- `--z_size FLOAT`: Camera frustum depth size (default: 0.3)
- `--keep_pct FLOAT`: Keep near percentile, discard far outliers (default: 95.0)
- `--min_depth FLOAT`: Minimum depth clipping in meters (default: 0.1)
- `--max_depth FLOAT`: Maximum depth clipping in meters, 0=no clip (default: 0.0)
- `--center`: Center point cloud at origin

### Examples

**Basic visualization:**
```bash
python vis_output_test.py --data_dir output_test_100
```

**Visualize first 50 frames with centering:**
```bash
python vis_output_test.py \
    --data_dir output_test_100 \
    --limit 50 \
    --center
```

**High-quality dense point cloud:**
```bash
python vis_output_test.py \
    --data_dir output_height20m \
    --stride 2 \
    --point_size 1.5 \
    --keep_pct 98 \
    --min_depth 0.5 \
    --max_depth 200 \
    --center
```

**Fast preview (sparse points):**
```bash
python vis_output_test.py \
    --data_dir output_height50m \
    --limit 20 \
    --stride 8 \
    --point_size 2.0
```

### Visualization Features

- **Point clouds**: Colored 3D points from depth maps
- **Camera frustums**: Shows camera poses and orientations
- **Camera images**: Displays RGB images at camera locations
- **Interactive**: Rotate, zoom, pan in browser (served by nerfvis)
- **OpenCV coordinates**: Uses OpenCV world coordinate system

---

## Complete Workflow

### Step 1: Convert Dataset

**Option A - Single scene:**
```bash
python convert_syndrone.py Town01_Opt_120/Town01_Opt_120/ClearNoon/height20m \
    --output_dir output_height20m
```

**Option B - Batch convert:**
```bash
./batch_convert.sh
```

### Step 2: Visualize Results

```bash
python vis_output_test.py \
    --data_dir output_height20m \
    --limit 50 \
    --stride 4 \
    --center
```

### Step 3: Verify Output

Check `complete_log.txt` in the output directory:
```bash
cat output_height20m/complete_log.txt
```

Example output:
```
Conversion completed at: 2025-10-04T12:34:56
Input: Town01_Opt_120/Town01_Opt_120/ClearNoon/height20m
Dataset: syndrone
Version: 0.1
Frames: 1234

Output format verification:
  - Depth: z-depth (perpendicular distance), NOT euclidean
  - Pose: world-to-camera (w2c) in OpenCV coordinate system
  - cam_from_worlds shape: (1234, 3, 4)
  - intrinsics shape: (1234, 3, 3)
```

---

## Troubleshooting

### Issue: "Cannot read RGB/depth/camera"
**Solution**: Ensure input directory has `rgb/`, `depth/`, `camera/` subdirectories with matching filenames.

### Issue: Visualization shows no points
**Solution**: Try adjusting `--min_depth`, `--max_depth`, and `--keep_pct` parameters.

### Issue: Point cloud is very sparse
**Solution**: Decrease `--stride` (e.g., from 4 to 2) for denser points.

### Issue: Visualization crashes or is slow
**Solution**: Increase `--stride` or use `--limit` to reduce data. Also check `--keep_pct` to filter outliers.

### Issue: "Already converted, skipping"
**Solution**: Remove `complete_log.txt` from output directory to force re-conversion.

---

## Technical Details

### Coordinate System Conversion

**Syndrone (Unreal Engine):**
- Left-handed: x=forward, y=right, z=up
- Depth: z-depth (uint16 PNG, scaled by 65535/1000)
- Pose: Rotator format (roll, pitch, yaw)

**Output (OpenCV):**
- Right-handed: x=right, y=down, z=forward
- Depth: z-depth (float32 NPY, meters)
- Pose: w2c matrix (3x4) [R|t]

### Depth Format

**Critical**: Syndrone depth is **z-depth** (perpendicular distance along optical axis), NOT euclidean (ray) distance. No conversion is needed, only scaling from uint16 to meters.

Formula: `depth_meters = depth_uint16 / 65535.0 * 1000.0`

### Camera Intrinsics

Computed from horizontal FOV:
```python
fx = (W * 0.5) / tan(fov * 0.5)
fy = (H * 0.5) / tan(fov_vertical * 0.5)
cx = (W - 1) / 2.0
cy = (H - 1) / 2.0
```
