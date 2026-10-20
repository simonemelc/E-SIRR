"""
Cross-Dataset Evaluation Script for Event-Guided SIRR (with R & T metrics)
=====================================================
Tests the pre-trained EventGuidedSIRR model on a new dataset (e.g., SIR2).
Features:
- Recursive nested directory scanning (loads T_gt, R_gt, I_mixed, Events).
- Dynamic cropping to multiples of 16.
- Metrics for both Background (T) and Reflection (R).
- 2-Row Visual grid saving for professional qualitative assessment.
- [NEW] Voxel density stratification: results split into SPARSE / MEDIUM / DENSE
  buckets based on the fraction of active voxel pixels (|E| > threshold).
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image as PILImage
import torchvision.transforms.functional as TF
import json
# Metrics
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity  as compute_ssim

# --- Project imports ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from model.ESIRR_Net import EventGuidedSIRR, SIRRLoss
from dataset.ESIRR_dataset import SIRRDataset, make_train_val_split

# LPIPS Import
try:
    import lpips
    have_lpips = True
except ImportError:
    print("LPIPS library not found. LPIPS metric will be skipped.")
    have_lpips = False


# ============================================================
#  Voxel Density Stratification
# ============================================================

# Thresholds that define the three density buckets.
# A pixel is considered "active" if its absolute voxel value exceeds ACTIVE_THRESH.
# The density score is the fraction of active pixels over the full voxel tensor.
#   density < SPARSE_THRESH            → SPARSE
#   SPARSE_THRESH <= density < DENSE_THRESH → MEDIUM
#   density >= DENSE_THRESH            → DENSE
ACTIVE_THRESH = 0.05   # minimum |voxel| value to count a pixel as active
SPARSE_THRESH = 0.010  # below this fraction of active pixels → sparse
DENSE_THRESH  = 0.030  # above this fraction of active pixels → dense

BUCKET_LABELS = ["sparse", "medium", "dense"]


def compute_voxel_density(e_voxel: torch.Tensor) -> float:
    """
    Compute the voxel density score for a single sample.

    Args:
        e_voxel: (B, bins, H, W) or (bins, H, W) — raw event voxel grid.

    Returns:
        density: float in [0, 1] — fraction of voxel entries with |val| > ACTIVE_THRESH.
    """
    if e_voxel.dim() == 4:
        e_voxel = e_voxel[0]  # remove batch dimension
    active = (torch.abs(e_voxel) > ACTIVE_THRESH).float()
    return active.mean().item()


def assign_bucket(density: float) -> str:
    """
    Map a scalar density score to one of the three bucket labels.

    Args:
        density: float output of compute_voxel_density().

    Returns:
        One of "sparse", "medium", or "dense".
    """
    if density < SPARSE_THRESH:
        return "sparse"
    elif density < DENSE_THRESH:
        return "medium"
    else:
        return "dense"


# ============================================================
#  Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluation for EventGuidedSIRR")

    # Path arguments
    p.add_argument("--data_root",  type=str, required=True,
                   help="Path to the dataset directory")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to the pre-trained .pth model weights")
    p.add_argument("--save_root",  type=str, default=".",
                   help="Where to save the output visualisations")

    # Evaluation settings
    p.add_argument("--dummy",    action="store_true",
                   help="Replace event voxel with zeros (ablation: no-event baseline)")
    p.add_argument("--padding",  action="store_true", default=False,
                   help="Pad images to multiples of 16 for U-Net compatibility")
    p.add_argument("--num_vis",  type=int, default=50,
                   help="Max visual grids to save")
    p.add_argument("--device",   type=str, default="cuda:0")
    p.add_argument("--dataset_type", type=str, default="sir2plus", 
                   choices=["sir2plus", "original"])
    p.add_argument("--gray", action="store_true", default=False,
                   help="Evaluate in grayscale mode")

    # --- Stratification ---
    p.add_argument("--sparse_thresh", type=float, default=SPARSE_THRESH,
                   help="Density score below which a sample is classified as SPARSE "
                        f"(default: {SPARSE_THRESH})")
    p.add_argument("--dense_thresh",  type=float, default=DENSE_THRESH,
                   help="Density score above which a sample is classified as DENSE "
                        f"(default: {DENSE_THRESH})")
    p.add_argument("--active_thresh", type=float, default=ACTIVE_THRESH,
                   help="Minimum |voxel| value for a pixel to be counted as active "
                        f"(default: {ACTIVE_THRESH})")

    return p.parse_args()

# ============================================================
#  Dataset
# ============================================================

class NestedCrossDataset(Dataset):
    def __init__(self, root_dir, dataset_type="sir2plus", gray_mode=False, padding=False):
        self.gray_mode   = gray_mode
        self.padding     = padding
        self.dataset_type = dataset_type
        self.samples     = []
        print(f"Scanning {dataset_type} directory recursively: {root_dir}...")

        if dataset_type == "sir2plus":
            for dirpath, _, filenames in os.walk(root_dir):
                for file in filenames:
                    if file.endswith("_I_mixed.jpg"):
                        prefix     = file.replace("_I_mixed.jpg", "")
                        mixed_path = os.path.join(dirpath, file)
                        voxel_path = os.path.join(dirpath, f"{prefix}_E_voxel.npy")
                        gt_t_path  = os.path.join(dirpath, f"{prefix}_T_gt.jpg")
                        gt_r_path  = os.path.join(dirpath, f"{prefix}_R_warped.jpg")

                        if (os.path.exists(voxel_path) and
                                os.path.exists(gt_t_path) and
                                os.path.exists(gt_r_path)):
                            self.samples.append({
                                "mixed": mixed_path,
                                "voxel": voxel_path,
                                "t_gt":  gt_t_path,
                                "r_gt":  gt_r_path,
                                "id":    prefix
                            })
        else:  # Original SIR2
            mixed_file = "mixture.png"
            valid_bg   = ["background.png", "groundtruth.png", "b.png",
                          "background.jpg", "groundtruth.jpg"]
            valid_refl = ["reflection.png", "r.png", "reflection.jpg"]
            for dirpath, _, filenames in os.walk(root_dir):
                if mixed_file in filenames:
                    matched_bg   = next((b for b in valid_bg   if b in filenames), None)
                    matched_refl = next((r for r in valid_refl if r in filenames), None)
                    if matched_bg:
                        rel_path = os.path.relpath(dirpath, root_dir).replace(os.sep, '_')
                        self.samples.append({
                            "mixed": os.path.join(dirpath, mixed_file),
                            "t_gt":  os.path.join(dirpath, matched_bg),
                            "voxel": None,
                            "r_gt":  os.path.join(dirpath, matched_refl) if matched_refl else None,
                            "id":    rel_path
                        })

        print(f"Found {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load RGB images (BGR -> RGB)
        img_mixed = cv2.cvtColor(cv2.imread(sample["mixed"]), cv2.COLOR_BGR2RGB)
        img_t_gt  = cv2.cvtColor(cv2.imread(sample["t_gt"]),  cv2.COLOR_BGR2RGB)

        # Resize GT to match mixture for original SIR2
        if self.dataset_type == "original" and img_t_gt.shape[:2] != img_mixed.shape[:2]:
            img_t_gt = cv2.resize(img_t_gt,
                                  (img_mixed.shape[1], img_mixed.shape[0]),
                                  interpolation=cv2.INTER_AREA)

        # Load R_gt — fall back to zeros if not available
        if sample["r_gt"] is not None and os.path.exists(sample["r_gt"]):
            img_r_gt = cv2.cvtColor(cv2.imread(sample["r_gt"]), cv2.COLOR_BGR2RGB)
        else:
            img_r_gt = np.zeros_like(img_mixed)

        # Load voxel — fall back to zeros for original SIR2
        if sample["voxel"] is not None and os.path.exists(sample["voxel"]):
            voxel = np.load(sample["voxel"])
        else:
            voxel = np.zeros((5, img_mixed.shape[0], img_mixed.shape[1]), dtype=np.float32)

        # Crop to multiple of 16 (unless padding mode is active)
        if not self.padding:
            H, W  = img_mixed.shape[:2]
            new_H = H - (H % 16)
            new_W = W - (W % 16)
            img_mixed = img_mixed[:new_H, :new_W, :]
            img_t_gt  = img_t_gt[:new_H,  :new_W, :]
            img_r_gt  = img_r_gt[:new_H,  :new_W, :]
            voxel     = voxel[:,            :new_H, :new_W]

        # Normalize and convert to tensors
        img_mixed = torch.from_numpy(img_mixed.transpose(2, 0, 1)).float() / 255.0
        img_t_gt  = torch.from_numpy(img_t_gt.transpose(2,  0, 1)).float() / 255.0
        img_r_gt  = torch.from_numpy(img_r_gt.transpose(2,  0, 1)).float() / 255.0
        voxel     = torch.from_numpy(voxel).float()

        if self.gray_mode:
            img_mixed = TF.rgb_to_grayscale(img_mixed, num_output_channels=3)
            img_t_gt  = TF.rgb_to_grayscale(img_t_gt,  num_output_channels=3)
            img_r_gt  = TF.rgb_to_grayscale(img_r_gt,  num_output_channels=3)

        return {
            "i_mixed":  img_mixed,
            "e_voxel":  voxel,
            "t_gt":     img_t_gt,
            "r_gt":     img_r_gt,
            "id":       sample["id"],
            "has_r_gt": sample["r_gt"] is not None,
        }


# ============================================================
#  Visualization utils
# ============================================================

def tensor_to_numpy_rgb(t):
    """Convert a (3, H, W) float tensor in [0,1] to a uint8 HxWx3 numpy array."""
    img = t.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def voxel_to_rgb_numpy_purple(voxel):
    """Render a voxel grid as a purple-tinted RGB image for visualization."""
    vox     = voxel.detach().cpu().numpy()
    max_pos = np.max(np.maximum( vox, 0), axis=0)
    max_neg = np.max(np.maximum(-vox, 0), axis=0)

    vox_rgb            = np.full((vox.shape[1], vox.shape[2], 3), [28, 3, 51], dtype=np.uint8)
    is_pos             = (max_pos > 0.01) & (max_pos >= max_neg)
    is_neg             = (max_neg > 0.01) & (max_neg >  max_pos)
    vox_rgb[is_pos]    = [223, 115, 255]
    vox_rgb[is_neg]    = [220, 220, 220]
    return vox_rgb


# ============================================================
#  Stratified results printer
# ============================================================

def print_stratified_results(bucket_metrics: dict, args):
    """
    Print per-bucket and overall aggregated evaluation metrics.

    bucket_metrics keys: "sparse", "medium", "dense", "all"
    Each value is a dict with lists: psnr_t, ssim_t, lpips_t, baseline_psnr,
    psnr_r, ssim_r.
    """
    print("\n" + "=" * 70)
    print(f"  STRATIFIED RESULTS  |  thresholds: "
          f"sparse<{args.sparse_thresh:.2f} / "
          f"dense>={args.dense_thresh:.2f} / "
          f"active|v|>{args.active_thresh:.2f}")
    print("=" * 70)

    for bucket in ["sparse", "medium", "dense", "all"]:
        m = bucket_metrics[bucket]
        n = len(m["psnr_t"])
        if n == 0:
            print(f"\n  [{bucket.upper():6s}]  No samples in this bucket.")
            continue

        avg_base   = np.mean(m["baseline_psnr"])
        avg_psnr_t = np.mean(m["psnr_t"])
        avg_ssim_t = np.mean(m["ssim_t"])
        avg_lpips  = np.mean(m["lpips_t"])
        improvement = avg_psnr_t - avg_base

        label = f"[{bucket.upper():6s}]  n={n:4d}"
        print(f"\n  {label}")
        print(f"    Baseline PSNR (Input)    : {avg_base:.2f} dB")
        print(f"    Model PSNR  (T_estimate) : {avg_psnr_t:.2f} dB  "
              f"({'+'if improvement>=0 else ''}{improvement:.2f} dB)")
        print(f"    Model SSIM  (T_estimate) : {avg_ssim_t:.4f}")
        print(f"    Model LPIPS (T_estimate) : {avg_lpips:.4f}")

        if len(m["psnr_r"]) > 0:
            print(f"    Model PSNR  (R_estimate) : {np.mean(m['psnr_r']):.2f} dB")
            print(f"    Model SSIM  (R_estimate) : {np.mean(m['ssim_r']):.4f}")

    print("\n" + "=" * 70)


# ============================================================
#  Main
# ============================================================

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Override module-level thresholds with CLI values so helpers use them
    global ACTIVE_THRESH, SPARSE_THRESH, DENSE_THRESH
    ACTIVE_THRESH = args.active_thresh
    SPARSE_THRESH = args.sparse_thresh
    DENSE_THRESH  = args.dense_thresh

    # Safety check: original SIR2 has no real events → must use --dummy
    if args.dataset_type == "original" and not args.dummy:
        print("\n[!] ERROR: You cannot test the 'original' dataset without --dummy "
              "because original SIR2 has no events. Add --dummy to the command.\n")
        sys.exit(1)


    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        return
    ckpt_name = Path(args.checkpoint).stem
    dummy_tag = "_dummy" if args.dummy else ""
    SAVE_VIS_DIR = os.path.join(args.save_root, f"results_{ckpt_name}{dummy_tag}")

    print(f"\nCkpt    : {args.checkpoint}")
    print(f"Data    : {args.data_root}")
    print(f"SaveDir : {SAVE_VIS_DIR}")
    print(f"Device  : {device}")


    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    train_args = checkpoint.get("args", {})


    print(f"\nDensity stratification thresholds:")
    print(f"  active pixel  |v| > {ACTIVE_THRESH}")
    print(f"  sparse  : density < {SPARSE_THRESH}")
    print(f"  medium  : {SPARSE_THRESH} <= density < {DENSE_THRESH}")
    print(f"  dense   : density >= {DENSE_THRESH}\n")

    model = EventGuidedSIRR(
        base       = train_args.get("base", 48),
        voxel_bins = train_args.get("voxel_bins", 5),
        ch_cap     = train_args.get("ch_cap", 192),
        norm       = train_args.get("norm", "gn"),
    ).to(device)

    model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
    model.eval()

    lpips_fn = lpips.LPIPS(net="alex", version="0.1").to(device)
    lpips_fn.eval()

    # --- Prepare dataset ---
    test_ds = NestedCrossDataset(
        args.data_root,
        dataset_type=args.dataset_type,
        gray_mode=args.gray,
        padding=args.padding,
    )
    if len(test_ds) == 0:
        print("No samples found. Exiting.")
        return

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    os.makedirs(SAVE_VIS_DIR, exist_ok=True)

    # --- Per-bucket metric accumulators ---
    # Each bucket stores lists of scalar metrics for all samples assigned to it.
    bucket_metrics = {
        bucket: defaultdict(list)
        for bucket in ["sparse", "medium", "dense", "all"]
    }

    # Per-bucket visualization counters (save up to num_vis per bucket)
    vis_counts = {"sparse": 0, "medium": 0, "dense": 0}

    print(f"\nStarting Cross-Dataset Evaluation on {args.dataset_type.upper()}...")

    bucket_ids_export = {"sparse": [], "medium": [], "dense": []}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            i_mixed   = batch["i_mixed"].to(device)
            e_voxel   = batch["e_voxel"].to(device)
            t_gt      = batch["t_gt"].to(device)
            r_gt      = batch["r_gt"].to(device)
            sample_id = batch["id"][0]
            has_r_gt  = batch["has_r_gt"][0].item()

            # Optionally zero-out events (ablation mode)
            events = torch.zeros_like(e_voxel) if args.dummy else e_voxel

            # ------------------------------------------------------------------
            # Compute voxel density and assign bucket BEFORE any padding,
            # so the density score reflects the raw voxel from disk.
            # ------------------------------------------------------------------
            density = compute_voxel_density(e_voxel)
            bucket  = assign_bucket(density)
            bucket_ids_export[bucket].append(sample_id)

            # --- Forward pass (with optional reflection padding) ---
            if args.padding:
                _, _, H, W = i_mixed.shape
                pad_h = (16 - (H % 16)) % 16
                pad_w = (16 - (W % 16)) % 16
                if pad_h > 0 or pad_w > 0:
                    i_mixed = torch.nn.functional.pad(i_mixed, (0, pad_w, 0, pad_h), mode="reflect")
                    events  = torch.nn.functional.pad(events,  (0, pad_w, 0, pad_h), mode="reflect")

                preds = model(i_mixed, events)
                t_est = preds["T_estimate"]
                r_est = preds["R_estimate"]

                if pad_h > 0 or pad_w > 0:
                    t_est   = t_est[:, :, :H, :W]
                    r_est   = r_est[:, :, :H, :W]
                    i_mixed = i_mixed[:, :, :H, :W]
            else:
                preds = model(i_mixed, events)
                t_est = preds["T_estimate"]
                r_est = preds["R_estimate"]

            # --- Convert to float numpy for metric computation ---
            gt_t_float  = tensor_to_numpy_rgb(t_gt[0]).astype(np.float32)  / 255.0
            est_t_float = tensor_to_numpy_rgb(t_est[0]).astype(np.float32) / 255.0
            gt_r_float  = tensor_to_numpy_rgb(r_gt[0]).astype(np.float32)  / 255.0
            est_r_float = tensor_to_numpy_rgb(r_est[0]).astype(np.float32) / 255.0
            mixed_float = tensor_to_numpy_rgb(i_mixed[0]).astype(np.float32) / 255.0

            # --- Background (T) metrics ---
            p_t    = compute_psnr(gt_t_float,  est_t_float, data_range=1.0)
            s_t    = compute_ssim(gt_t_float,  est_t_float, channel_axis=-1, data_range=1.0)
            p_base = compute_psnr(gt_t_float,  mixed_float, data_range=1.0)

            est_np = (est_t_float * 255).astype(np.uint8)
            gt_np  = (gt_t_float  * 255).astype(np.uint8)

            lpips_val = lpips_fn(
                (t_est * 2.0 - 1.0), 
                (t_gt * 2.0 - 1.0)
            ).item()

            # --- Reflection (R) metrics — only when R_gt is available ---
            p_r, s_r = None, None
            if has_r_gt:
                p_r = compute_psnr(gt_r_float, est_r_float, data_range=1.0)
                s_r = compute_ssim(gt_r_float, est_r_float, channel_axis=-1, data_range=1.0)

            # --- Accumulate into the sample's bucket AND the global "all" bucket ---
            for target_bucket in [bucket, "all"]:
                m = bucket_metrics[target_bucket]
                m["psnr_t"].append(p_t)
                m["ssim_t"].append(s_t)
                m["baseline_psnr"].append(p_base)
                m["lpips_t"].append(lpips_val)
                m["density"].append(density)
                if p_r is not None:
                    m["psnr_r"].append(p_r)
                    m["ssim_r"].append(s_r)

            # ------------------------------------------------------------------
            # Visualization — save up to num_vis images per bucket so that
            # the qualitative grid covers all three density regimes equally.
            # The bucket label and density score are embedded in the filename.
            # ------------------------------------------------------------------
            if vis_counts[bucket] < args.num_vis:
                gt_t_np  = (gt_t_float  * 255).astype(np.uint8)
                est_t_np = (est_t_float * 255).astype(np.uint8)
                gt_r_np  = (gt_r_float  * 255).astype(np.uint8)
                est_r_np = (est_r_float * 255).astype(np.uint8)
                mixed_np = (mixed_float * 255).astype(np.uint8)
                ev_np    = voxel_to_rgb_numpy_purple(e_voxel[0])
                black_np = np.zeros_like(mixed_np)

                # Row 1: Mixed | Events | T_est | T_gt
                row1 = np.concatenate([mixed_np, ev_np, est_t_np, gt_t_np], axis=1)
                # Row 2: Empty  | Empty   | R_est | R_gt
                row2 = np.concatenate([black_np, black_np, est_r_np, gt_r_np], axis=1)

                concat_img = np.concatenate([row1, row2], axis=0)

                # Embed bucket label and density in filename for easy filtering
                save_path = os.path.join(
                    SAVE_VIS_DIR,
                    f"{bucket}_d{density:.3f}_{sample_id}_psnrT{p_t:.1f}.png"
                )
                PILImage.fromarray(concat_img).save(save_path)
                vis_counts[bucket] += 1

    # --- Print overall and per-bucket results ---
    print_stratified_results(bucket_metrics, args)

    # --- Print bucket sample counts for quick sanity check ---
    print("\n  Sample distribution across buckets:")
    for b in ["sparse", "medium", "dense"]:
        n = len(bucket_metrics[b]["psnr_t"])
        total = len(bucket_metrics["all"]["psnr_t"])
        pct = 100.0 * n / total if total > 0 else 0.0
        avg_d = np.mean(bucket_metrics[b]["density"]) if n > 0 else 0.0
        print(f"    {b:6s}: {n:4d} samples ({pct:5.1f}%)  avg density={avg_d:.4f}")
    print(f"\n  Visuals saved in: {SAVE_VIS_DIR}/\n")

    split_path = os.path.join(SAVE_VIS_DIR, f"bucket_splits_{args.dataset_type}.json")
    with open(split_path, "w") as f:
        json.dump(bucket_ids_export, f, indent=4)
    print(f"Bucket splits saved in: {split_path}")


if __name__ == "__main__":
    main()

"""
python test_SIR2+.py \
  --data_root /medias/db/ImagingSecurity_misc/melcarne/E-SIRR/utils/ER-100_RealDVS_Dataset/processed \
  --checkpoint /medias/db/ImagingSecurity_misc/melcarne/Events_SIRR/outputs/train/coco_coco_blend_05_cross_stage_gray/C48_cap192_e2e_lr0.0001_bs4_wG_0.0_wExcl_0.02_conf_map_cross_stage_gray/checkpoints/best_model.pth \
  --device cuda:0 
"""