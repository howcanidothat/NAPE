import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
import json
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from torchreid.scripts.builder import build_config
    from torchreid.utils.tools import read_keypoints
    from torchreid.data.masks_transforms import CocoToEightBodyMasks
    from torchreid.data.datasets.keypoints_to_masks import rescale_keypoints
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    print("Warning: torchreid not available, using standalone mode")
    
    # Standalone implementations
    def read_keypoints(path):
        """Read keypoints from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return np.array(data)
        if 'keypoints' in data:
            return np.array(data['keypoints'])
        return np.array(data)
    
    def rescale_keypoints(kp, src_size, dst_size):
        """Rescale keypoints to new image size."""
        kp = kp.copy()
        scale_x = dst_size[0] / src_size[0]
        scale_y = dst_size[1] / src_size[1]
        kp[:, 0] *= scale_x
        kp[:, 1] *= scale_y
        return kp
    
    class CocoToEightBodyMasks:
        """Simplified version for standalone mode."""
        def __init__(self):
            # COCO keypoint to body part mapping (0-7 parts)
            # Head: 0-4, Torso: 5-6, Arms: 7-10, Legs: 11-14, Feet: 15-16
            self.kp_to_part = [
                0, 0, 0, 0, 0,  # nose, eyes, ears -> Head
                1, 1,  # shoulders -> Torso
                2, 2, 2, 2,  # arms -> Arms
                3, 3,  # hips -> Legs
                3, 3,  # knees -> Legs
                4, 4   # ankles -> Feet
            ]
        
        def apply_to_keypoints_xyc(self, kp_xyc):
            """Add part group index to keypoints."""
            n_kp = len(kp_xyc)
            kp_xyck = np.zeros((n_kp, 4))
            kp_xyck[:, :3] = kp_xyc
            for i in range(min(n_kp, 17)):
                kp_xyck[i, 3] = self.kp_to_part[i]
            return kp_xyck
    
    def build_config(config_path=None):
        """Dummy config for standalone mode."""
        class DummyConfig:
            class data:
                width = 128
                height = 256
            class model:
                class kpr:
                    class keypoints:
                        kp_dir = "pifpaf_keypoints_pifpaf_maskrcnn_filtering"
        return DummyConfig()

import matplotlib.cm as cm

def scale_lightness(rgb, scale_l=1.4):
    import colorsys
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)

def draw_keypoints_with_parts(img, kp, model_img_size=None, radius=4, thickness=-1, vis_thresh=0.2):
    """
    Draw keypoints on the image, colored by their body part group (Head, Torso, Arms, Legs, Feet).
    If img is None, creates a blank canvas based on keypoint bounding box.
    """
    if img is None:
        # Create canvas based on keypoint extent
        valid_kp = kp[kp[:, 2] > vis_thresh]
        if len(valid_kp) == 0:
            img_h, img_w = 256, 128  # Default size
        else:
            margin = 20
            min_x, max_x = valid_kp[:, 0].min(), valid_kp[:, 0].max()
            min_y, max_y = valid_kp[:, 1].min(), valid_kp[:, 1].max()
            img_w = int(max_x - min_x + 2 * margin)
            img_h = int(max_y - min_y + 2 * margin)
            # Adjust keypoints to fit in new canvas
            kp = kp.copy()
            kp[:, 0] -= (min_x - margin)
            kp[:, 1] -= (min_y - margin)
        img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255  # White background
    
    if model_img_size is not None:
        kp = rescale_keypoints(kp, model_img_size, (img.shape[1], img.shape[0]))
    
    # Use the grouping logic from the project
    # If the model uses 5 parts, we map COCO joints to these 5 groups.
    # The image provided shows 5 distinct colors.
    
    # Typical mapping for 5 parts (Head, Torso, Arms, Legs, Feet):
    # This might vary, but we'll use a standard mapping or the one from CocoToEightBodyMasks if available.
    # For now, let's use a simpler mapping based on COCO indices:
    # 0-4: Head (Nose, Eyes, Ears)
    # 5-6: Shoulders (Torso/Arms)
    # 7-10: Arms
    # 11-12: Hips (Torso/Legs)
    # 13-16: Legs/Feet
    
    # Let's try to use the built-in grouping if possible
    grouper = CocoToEightBodyMasks()
    # xyck: x, y, confidence, k (group index)
    kp_grouped = grouper.apply_to_keypoints_xyc(kp)
    
    max_k = int(kp_grouped[:, -1].max())
    
    for xyck in kp_grouped:
        x, y, c, k = xyck
        if c > vis_thresh:
            # Color based on group k
            color_raw = scale_lightness(cm.gist_rainbow(k / max(1, max_k))[0:-1])
            color = (np.array(color_raw) * 255).astype(np.uint8)
            color_bgr = (int(color[2]), int(color[1]), int(color[0]))
            
            cv2.circle(
                img,
                (int(x), int(y)),
                radius=radius,
                color=color_bgr,
                thickness=thickness, # Filled circle
                lineType=cv2.LINE_AA
            )
    return img

def _collect_images(root: Path, dataset: str, split: str, limit: int) -> list[Path]:
    dataset_dir = root / dataset
    split_dir_map = {
        "train": "bounding_box_train",
        "query": "query",
        "gallery": "bounding_box_test",
        "test": "bounding_box_test",
        "all": ["bounding_box_train", "query", "bounding_box_test"]
    }
    
    img_paths = []
    
    target_splits = split_dir_map.get(split, [split])
    if isinstance(target_splits, str):
        target_splits = [target_splits]
        
    for s in target_splits:
        split_dir = dataset_dir / s

        # Occluded_REID structure
        if (dataset_dir / "occluded_body_images").is_dir() and (dataset_dir / "whole_body_images").is_dir():
            if s in ["bounding_box_train", "query", "train"]:
                split_dir = dataset_dir / "occluded_body_images"
            else:
                split_dir = dataset_dir / "whole_body_images"

        if not split_dir.is_dir() and dataset_dir.is_dir():
            # If specified split dir doesn't exist, search entire dataset dir
            split_dir = dataset_dir

        if split_dir.is_dir():
            # First try direct image search
            found = list(split_dir.rglob("*.jpg")) + list(split_dir.rglob("*.png")) + \
                    list(split_dir.rglob("*.jpeg")) + list(split_dir.rglob("*.tif"))
            img_paths.extend(found)
    
    # If no images found in standard directories, try external_annotation and masks
    if not img_paths:
        for subdir in ["external_annotation", "masks"]:
            subdir_path = dataset_dir / subdir
            if subdir_path.is_dir():
                for split_sub in ["train", "query", "gallery", "bounding_box_train", "bounding_box_test"]:
                    full_path = subdir_path / subdir_path.name if (subdir_path / subdir_path.name).is_dir() else subdir_path
                    # Try to find image-like files in subdirectories
                    for ext in ["*.jpg", "*.png", "*.jpeg", "*.tif"]:
                        img_paths.extend(list(full_path.rglob(ext)))
    
    # Also check if images are in the dataset root with specific patterns
    if not img_paths:
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.tif"]:
            img_paths.extend(list(dataset_dir.rglob(ext)))
    
    img_paths = sorted(list(set(img_paths))) # Remove duplicates and sort
    if limit > 0:
        img_paths = img_paths[:limit]
    return img_paths

def _collect_keypoints(root: Path, dataset: str, split: str, limit: int, kp_dir_name: str = None) -> list[Path]:
    """Collect keypoint JSON files directly from external_annotation directory."""
    dataset_dir = root / dataset
    kp_paths = []
    
    # Check external_annotation directory
    ext_ann_dir = dataset_dir / "external_annotation"
    if ext_ann_dir.is_dir():
        # Search all subdirectories recursively for JSON files
        if kp_dir_name:
            # If kp_dir_name specified, look for it
            target_dir = ext_ann_dir / kp_dir_name
            if target_dir.is_dir():
                # Search in all split subdirectories
                for split_sub in [split, "train", "query", "gallery", "bounding_box_train", "bounding_box_test"]:
                    split_dir = target_dir / split_sub
                    if split_dir.is_dir():
                        kp_paths.extend(list(split_dir.glob("*.json")))
        
        # If still no results, search all JSON files recursively
        if not kp_paths:
            kp_paths = list(ext_ann_dir.rglob("*.json"))
    
    kp_paths = sorted(list(set(kp_paths)))
    if limit > 0:
        kp_paths = kp_paths[:limit]
    return kp_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="query")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--filenames", type=str, nargs="+", help="Specific filenames to process")
    parser.add_argument("--out-dir", type=str, default="assets/keypoint_overlays")
    parser.add_argument("--vis-thresh", type=float, default=0.2)
    parser.add_argument("--from-kp-only", action="store_true", help="Generate from keypoint JSON files only (no original images)")
    args = parser.parse_args()

    cfg = build_config(config_path=args.config_file)
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    kp_dir_name = getattr(cfg.model.kpr.keypoints, "kp_dir", None)

    if args.from_kp_only:
        # Directly collect keypoint JSON files
        kp_paths = _collect_keypoints(root, args.dataset, args.split, args.limit, kp_dir_name)
        
        print(f"Total keypoint files found: {len(kp_paths)}")
        if kp_paths:
            print(f"First few files: {[p.name for p in kp_paths[:3]]}")
        
        if args.filenames:
            filtered_paths = []
            for p in kp_paths:
                # Check if any specified filename is contained in the path
                if any(f in p.name or f in str(p) for f in args.filenames):
                    filtered_paths.append(p)
            kp_paths = filtered_paths
            
        print(f"Found {len(kp_paths)} keypoint files for {args.dataset} {args.split}")
        
        count = 0
        for kp_path in kp_paths:
            kps_all = read_keypoints(str(kp_path))
            if kps_all is None or len(kps_all) == 0:
                continue
            # Handle different keypoint formats
            kps = kps_all[0] if kps_all.ndim == 3 else kps_all
            kps = np.asarray(kps).reshape((-1, 3))
            
            # Draw keypoints on blank canvas
            img_kp = draw_keypoints_with_parts(None, kps, model_img_size=None, vis_thresh=args.vis_thresh)
            
            out_path = out_dir / f"{args.dataset}_{kp_path.stem}_kp.jpg"
            cv2.imwrite(str(out_path), img_kp)
            count += 1
        
        print(f"Saved {count} keypoint overlays to {out_dir}")
    else:
        # Original image-based workflow
        img_paths = _collect_images(root, args.dataset, args.split, args.limit)
        
        if args.filenames:
            filtered_paths = []
            for p in img_paths:
                if any(f in p.name for f in args.filenames):
                    filtered_paths.append(p)
            img_paths = filtered_paths
            
        print(f"Found {len(img_paths)} images for {args.dataset} {args.split}")

        count = 0
        for img_path in img_paths:
            kp_path = _infer_kp_path(cfg, root / args.dataset, args.split, img_path)
            if kp_path is None or not kp_path.exists():
                continue
                
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            kps_all = read_keypoints(str(kp_path))
            # Handle different keypoint formats
            kps = kps_all[0] if kps_all.ndim == 3 else kps_all
            kps = np.asarray(kps).reshape((-1, 3))
            
            # Draw keypoints colored by part
            model_size = [cfg.data.width, cfg.data.height]
            img_kp = draw_keypoints_with_parts(img.copy(), kps, model_img_size=model_size, vis_thresh=args.vis_thresh)
            
            out_path = out_dir / f"{args.dataset}_{img_path.stem}_kp.jpg"
            cv2.imwrite(str(out_path), img_kp)
            count += 1

        print(f"Saved {count} keypoint overlays to {out_dir}")

if __name__ == "__main__":
    main()
