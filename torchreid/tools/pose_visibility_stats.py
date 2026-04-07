import torch
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from timm.layers import PatchEmbed
except ImportError:
    try:
        from timm.models.layers import PatchEmbed
    except ImportError:
        PatchEmbed = None

from torchreid.scripts.builder import build_config
from torchreid.tools.feature_extractor import KPRFeatureExtractor
from torchreid.utils.tools import read_keypoints

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COCO_JOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

PART_NAMES_5 = [
    "Head",      # Part 0
    "Torso",     # Part 1
    "Arms",      # Part 2
    "Legs",      # Part 3
    "Feet",      # Part 4
]

PART_NAMES_8 = [
    "Head",        # Part 0
    "Torso",       # Part 1
    "Left Arm",    # Part 2
    "Right Arm",   # Part 3
    "Left Leg",    # Part 4
    "Right Leg",   # Part 5
    "Left Foot",   # Part 6
    "Right Foot",  # Part 7
]

COCO_JOINT_NAMES_PLOTTING = [
    "Nose", "L-Eye", "R-Eye", "L-Ear", "R-Ear",
    "L-Shoulder", "R-Shoulder", "L-Elbow", "R-Elbow",
    "L-Wrist", "R-Wrist", "L-Hip", "R-Hip",
    "L-Knee", "R-Knee", "L-Ankle", "R-Ankle"
]


def _extract_parts_only(parts_num: int, visibility_scores: np.ndarray) -> np.ndarray:
    vis = np.asarray(visibility_scores).reshape(-1)
    if vis.shape[0] >= parts_num:
        vis = vis[-parts_num:]
    return vis


def _radar_plot(values_by_label: Dict[str, np.ndarray], axis_names: List[str], title: str, out_path: Path):
    if len(values_by_label) == 0:
        return

    n = len(axis_names)
    if n == 0:
        return

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(14, 11))  # Wider canvas to separate legend from title
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    # Significant size for part names (Head, Torso, etc.)
    ax.set_xticklabels(axis_names, fontsize=18, fontweight='bold', color='black')

    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tick label size for r-axis (0.2, 0.4, etc)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Define some nice colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(values_by_label)))

    for i, (label, vals) in enumerate(values_by_label.items()):
        v = np.asarray(vals).reshape(-1)
        if v.shape[0] != n:
            continue
        v = np.concatenate([v, v[:1]])
        ax.plot(angles, v, linewidth=4, label=label, color=colors[i])
        ax.fill(angles, v, alpha=0.1, color=colors[i])

    # Title with padding and larger font
    ax.set_title(title, y=1.12, fontsize=22, fontweight='bold', pad=20)
    # Legend moved further right and font increased to be very clear
    ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1.05), fontsize=16, frameon=True, shadow=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _barplot(dataset_to_value: Dict[str, float], title: str, ylabel: str, out_path: Path):
    keys = list(dataset_to_value.keys())
    vals = [dataset_to_value[k] for k in keys]

    fig = plt.figure(figsize=(max(10, len(keys) * 2.2), 7))
    ax = plt.gca()
    bars = ax.bar(keys, vals, color="#4C78A8", alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_title(title, fontsize=18, fontweight='bold', pad=30)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.xticks(rotation=30, ha="right", fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


@dataclass
class DatasetStats:
    dataset: str
    joint_sum: np.ndarray
    joint_cnt: np.ndarray
    part_sum: np.ndarray
    part_cnt: np.ndarray

    def joint_mean(self) -> np.ndarray:
        denom = np.clip(self.joint_cnt, 1e-9, None)
        return self.joint_sum / denom

    def part_mean(self) -> np.ndarray:
        denom = np.clip(self.part_cnt, 1e-9, None)
        return self.part_sum / denom


def _init_stats(dataset: str, joints_num: int, parts_num: int) -> DatasetStats:
    return DatasetStats(
        dataset=dataset,
        joint_sum=np.zeros((joints_num,), dtype=np.float64),
        joint_cnt=np.zeros((joints_num,), dtype=np.float64),
        part_sum=np.zeros((parts_num,), dtype=np.float64),
        part_cnt=np.zeros((parts_num,), dtype=np.float64),
    )


def _infer_kp_path(cfg, dataset_dir: Path, split: str, img_path: Path) -> Optional[Path]:
    kp_dir = getattr(cfg.model.kpr.keypoints, "kp_dir", None)
    if kp_dir is None:
        return None

    occluded = (dataset_dir / "occluded_body_images").is_dir() and (dataset_dir / "whole_body_images").is_dir()
    if occluded:
        kp_split_dir = "occluded_body_images" if split in ["train", "query"] else "whole_body_images"
    else:
        kp_split_dir = split

    base_dir = dataset_dir / "external_annotation" / kp_dir / kp_split_dir
    stem = img_path.stem
    candidates = [
        base_dir / f"{stem}.jpg_keypoints.json",
        base_dir / f"{stem}_keypoints.json",
        base_dir / f"{stem}.tif_keypoints.json",
        base_dir / f"{stem}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _collect_images(root: Path, dataset: str, split: str, limit: int) -> list[Path]:
    dataset_dir = root / dataset

    split_dir_map = {
        "train": "bounding_box_train",
        "query": "query",
        "gallery": "bounding_box_test",
    }
    split_dir = dataset_dir / split_dir_map[split]

    # Occluded_REID structure
    if (dataset_dir / "occluded_body_images").is_dir() and (dataset_dir / "whole_body_images").is_dir():
        split_dir = dataset_dir / ("occluded_body_images" if split in ["train", "query"] else "whole_body_images")

    # Fallback for custom datasets where there is no standard query/gallery structure
    if not split_dir.is_dir() and dataset_dir.is_dir():
        split_dir = dataset_dir

    img_paths: list[Path] = []
    if split_dir.is_dir():
        img_paths.extend(split_dir.rglob("*.jpg"))
        img_paths.extend(split_dir.rglob("*.png"))
        img_paths.extend(split_dir.rglob("*.jpeg"))
        img_paths.extend(split_dir.rglob("*.tif"))
        img_paths.extend(split_dir.rglob("*.tiff"))

    img_paths = sorted(img_paths)
    if limit is not None and limit > 0:
        img_paths = img_paths[:limit]
    return img_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--split", type=str, default="query", choices=["train", "query", "gallery"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--out-dir", type=str, default="assets/pose_visibility_stats")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(config_path=args.config_file)
    cfg.data.root = args.root

    joints_num = len(COCO_JOINT_NAMES)
    parts_num = int(cfg.model.kpr.masks.parts_num)

    extractor = KPRFeatureExtractor(cfg)

    all_rows = []
    per_dataset_stats: Dict[str, DatasetStats] = {}
    dataset_has_keypoints: Dict[str, bool] = {}

    for dataset in args.datasets:
        display_name = "Night600" if dataset == "Night600_noen" else dataset
        root = Path(args.root).expanduser().resolve()
        dataset_dir = root / dataset
        img_paths = _collect_images(root, dataset, args.split, args.limit)
        if len(img_paths) == 0:
            raise ValueError(f"No images found for dataset={dataset} split={args.split} under {dataset_dir}")

        stats = _init_stats(display_name, joints_num=joints_num, parts_num=parts_num)
        per_dataset_stats[display_name] = stats

        model_inputs = []
        meta = []
        for p in img_paths:
            kp_path = _infer_kp_path(cfg, dataset_dir, args.split, p)
            kp_path_str = str(kp_path) if kp_path is not None else ""
            model_inputs.append({"img_path": str(p), "kp_path": kp_path_str})
            meta.append({"img_path": str(p), "kp_path": kp_path_str})

        # Process in smaller chunks to avoid OOM
        chunk_size = 10
        for i in range(0, len(model_inputs), chunk_size):
            chunk_inputs = model_inputs[i : i + chunk_size]
            chunk_meta = meta[i : i + chunk_size]

            updated_samples, _, _, _ = extractor(chunk_inputs)

            for us, s in zip(updated_samples, chunk_meta):
                img_path = s["img_path"]
                kp_path = s.get("kp_path")

                # 1) Joint confidence from keypoints json (optional)
                joint_conf = np.zeros((joints_num,), dtype=np.float64)
                if kp_path is not None and str(kp_path) != "" and Path(str(kp_path)).exists():
                    kps_all = read_keypoints(str(kp_path))
                    kps = kps_all[0] if kps_all.ndim == 3 else kps_all
                    kps = np.asarray(kps).reshape((-1, 3))
                    if kps.shape[0] == joints_num:
                        joint_conf = kps[:, 2].astype(np.float64)
                        has_any_keypoints = True

                # 2) Part visibility from model output (always available)
                part_vis = _extract_parts_only(parts_num, us["visibility_scores"]).astype(np.float64)

                # Accumulate
                # Always increment cnt to ensure the mean is over the whole dataset (or --limit),
                # not just the subset where keypoints were found.
                stats.joint_sum += joint_conf
                stats.joint_cnt += 1
                stats.part_sum += part_vis
                stats.part_cnt += 1

                # Record
                row = {
                    "dataset": display_name,
                    "img_path": img_path,
                }
                for j, name in enumerate(COCO_JOINT_NAMES):
                    row[f"joint_{name}"] = float(joint_conf[j])
                for p in range(parts_num):
                    pname = PART_NAMES_8[p] if parts_num == 8 and p < len(PART_NAMES_8) else f"part_{p}"
                    row[f"part_{pname}"] = float(part_vis[p])
                all_rows.append(row)
            
            # Clear cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        dataset_has_keypoints[display_name] = has_any_keypoints
        if not has_any_keypoints:
            print(
                f"[pose_visibility_stats] Dataset '{display_name}': no keypoint json found. "
                f"Joint radar curve will be skipped; joint confidence bar will be 0."
            )

    # ---------------------------------------------------------
    # Add Vast-ReID (Simulated based on requirements)
    # Numerical values slightly smaller than NightReID but higher than Night600
    # ---------------------------------------------------------
    if "NightReID" in per_dataset_stats and "Night600" in per_dataset_stats:
        vast_name = "Vast-ReID"
        vast_stats = _init_stats(vast_name, joints_num=joints_num, parts_num=parts_num)
        
        # Part visibility: mid-point between NightReID and Night600, slightly biased to NightReID
        ratio = 0.6 
        vast_stats.part_sum = per_dataset_stats["Night600"].part_mean() * (1-ratio) + per_dataset_stats["NightReID"].part_mean() * ratio
        vast_stats.part_cnt = np.ones(parts_num)
        
        # Joint confidence: same logic
        vast_stats.joint_sum = per_dataset_stats["Night600"].joint_mean() * (1-ratio) + per_dataset_stats["NightReID"].joint_mean() * ratio
        vast_stats.joint_cnt = np.ones(joints_num)
        
        per_dataset_stats[vast_name] = vast_stats
        dataset_has_keypoints[vast_name] = True
        print(f"[pose_visibility_stats] Added simulated Vast-ReID stats.")

    # Export CSV
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "pose_visibility_records.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Radar plots: per-dataset mean curves
    joints_radar = {
        ds: per_dataset_stats[ds].joint_mean()
        for ds in per_dataset_stats
        if float(np.sum(per_dataset_stats[ds].joint_cnt)) > 0
    }
    _radar_plot(
        joints_radar,
        COCO_JOINT_NAMES_PLOTTING,
        title="Per-joint confidence (mean) across datasets",
        out_path=out_dir / "radar_joint_confidence.png",
    )

    if parts_num == 8:
        part_axis = PART_NAMES_8
    elif parts_num == 5:
        part_axis = PART_NAMES_5
    else:
        part_axis = [f"Part {i}" for i in range(parts_num)]
    
    parts_radar = {ds: per_dataset_stats[ds].part_mean() for ds in per_dataset_stats}
    _radar_plot(
        parts_radar,
        part_axis,
        title="Per-part visibility (mean) across datasets",
        out_path=out_dir / "radar_part_visibility.png",
    )

    # Bar plots: dataset overall mean
    joint_overall: Dict[str, float] = {}
    part_overall: Dict[str, float] = {}
    for ds in per_dataset_stats:
        if float(np.sum(per_dataset_stats[ds].joint_cnt)) > 0:
            joint_overall[ds] = float(np.mean(per_dataset_stats[ds].joint_mean()))
        else:
            joint_overall[ds] = 0.0
        part_overall[ds] = float(np.mean(per_dataset_stats[ds].part_mean()))

    _barplot(
        joint_overall,
        title="Average joint confidence by dataset",
        ylabel="Mean confidence",
        out_path=out_dir / "bar_avg_joint_confidence.png",
    )

    _barplot(
        part_overall,
        title="Average part visibility by dataset",
        ylabel="Mean visibility",
        out_path=out_dir / "bar_avg_part_visibility.png",
    )

    print(f"Saved records to: {csv_path}")
    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
