# Coolant Dataset Tools

Utilities for converting and visualising the Coolant matcher dataset in the
unified format used by VGGt workflows. The layout mirrors
`convert_syndrone2vggt/` to keep command usage familiar.

## 1. Convert to Unified Format

Script: `convert_coolant.py`

```bash
python convert_coolant.py \
    /workspace/Coolant_datset/matcher_datasets/Coolant/ArkansasExperiment/Sept2025/\
    sparse_h80_so0.7_fo0.8_pit82_62 --limit 0
```

Key details:

- Detects nested scene folders automatically (handles `<scene>/<scene>`).
- Copies RGB images, preserves float32 z-depth (meters), converts poses from
  camera-to-world to world-to-camera (`cam_from_worlds.npy`).
- Produces sequential filenames (`images/00000.jpg`, `depths/00000.npy`), plus
  `intrinsics.npy` and `image_names.json`.

## 2. Visualise with Nerfvis

Script: `coolant_multi_frame_vis.py`

Works with either the raw Coolant scene (`images/`, `raw_depths/`, `poses/`) or
the converted output directory.

```bash
python coolant_multi_frame_vis.py --data_dir path/to/converted_scene --stride 4
```

Options:

- `--limit` to view a subset.
- `--stride` to subsample depth points.
- `--keep_pct`, `--min_depth`, `--max_depth` to control point filtering.
- `--global_extrinsic` to apply an additional 4x4 transform (optional).

The viewer shows camera frustums, billboards, and coloured depth point clouds
using OpenCV coordinate conventions.


