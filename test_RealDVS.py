"""
RealDVS Evaluation Script for Event-Guided SIRR
================================================
Tests the pre-trained EventGuidedSIRR model on the RealDVS dataset.

Dataset structure (--data_root):
    <data_root>/
        scene01/
            chessboard_f0814_I_mixed.jpg
            chessboard_f0814_E_voxel.npy
            chessboard_f0814_E_sum.png
            GT/
                chessboard_GT.jpg
        scene02/
            ...

Features:
  - Dynamically pads spatial dimensions to multiples of 16 for U-Net compatibility.
  - Loads frames in true single-channel grayscale (cv2.IMREAD_GRAYSCALE) to avoid 
    JPEG chroma subsampling discrepancies.
  - Computes PSNR, SSIM, and LPIPS directly on GPU tensors where possible.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image as PILImage

# Metrics
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# --- Project imports ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from model.ESIRR_Net import EventGuidedSIRR

# LPIPS
try:
    import lpips
    have_lpips = True
except ImportError:
    print("LPIPS library not found. LPIPS metric will be skipped.")
    have_lpips = False


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

    return p.parse_args()



# ============================================================
#  Dataset
# ============================================================

class RealDVSDataset(Dataset):
    def __init__(self, root_dir: str, padding: bool = True):
        self.padding   = padding
        self.samples   = []
        root_dir       = Path(root_dir)

        print(f"Scanning RealDVS directory: {root_dir} ...")

        for scene_dir in sorted(root_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            gt_dir = scene_dir / "GT"
            if not gt_dir.exists():
                continue

            gt_candidates = list(gt_dir.glob("*_GT.jpg")) + list(gt_dir.glob("*_GT.png"))
            if not gt_candidates:
                continue
            gt_path = gt_candidates[0]
            scene_prefix = gt_path.stem.replace("_GT", "")

            for mixed_path in sorted(scene_dir.glob(f"{scene_prefix}_f*_I_mixed.jpg")):
                frame_token = mixed_path.stem.replace("_I_mixed", "")
                voxel_path = scene_dir / f"{frame_token}_E_voxel.npy"
                esum_path  = scene_dir / f"{frame_token}_E_sum.png"

                if not voxel_path.exists():
                    continue

                self.samples.append({
                    "mixed":  str(mixed_path),
                    "voxel":  str(voxel_path),
                    "e_sum":  str(esum_path) if esum_path.exists() else None,
                    "t_gt":   str(gt_path),
                    "id":     f"{scene_dir.name}_{frame_token}",
                    "scene":  scene_dir.name,
                })

        print(f"Found {len(self.samples)} valid frame samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load as native single-channel grayscale
        img_mixed = cv2.imread(s["mixed"], cv2.IMREAD_GRAYSCALE)
        img_t_gt  = cv2.imread(s["t_gt"],  cv2.IMREAD_GRAYSCALE)

        if img_t_gt.shape != img_mixed.shape:
            img_t_gt = cv2.resize(img_t_gt, (img_mixed.shape[1], img_mixed.shape[0]), interpolation=cv2.INTER_AREA)

        if s["e_sum"] is not None:
            e_sum_img = cv2.cvtColor(cv2.imread(s["e_sum"]), cv2.COLOR_BGR2RGB)
            if e_sum_img.shape[:2] != img_mixed.shape:
                e_sum_img = cv2.resize(e_sum_img, (img_mixed.shape[1], img_mixed.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            e_sum_img = np.zeros((img_mixed.shape[0], img_mixed.shape[1], 3), dtype=np.uint8)

        voxel = np.load(s["voxel"]).astype(np.float32)
        if voxel.shape[1:] != img_mixed.shape:
            new_bins = [cv2.resize(voxel[b], (img_mixed.shape[1], img_mixed.shape[0]), interpolation=cv2.INTER_LINEAR) for b in range(voxel.shape[0])]
            voxel = np.stack(new_bins, axis=0)

        if not self.padding:
            H, W  = img_mixed.shape
            new_H = H - (H % 16)
            new_W = W - (W % 16)
            img_mixed = img_mixed[:new_H, :new_W]
            img_t_gt  = img_t_gt[:new_H,  :new_W]
            e_sum_img = e_sum_img[:new_H,  :new_W, :]
            voxel     = voxel[:, :new_H, :new_W]

        img_mixed = torch.from_numpy(img_mixed[np.newaxis]).float() / 255.0
        img_t_gt  = torch.from_numpy(img_t_gt[np.newaxis]).float()  / 255.0
        voxel     = torch.from_numpy(voxel).float()

        return {
            "i_mixed":  img_mixed,
            "e_voxel":  voxel,
            "t_gt":     img_t_gt,
            "e_sum_np": e_sum_img,
            "id":       s["id"],
            "scene":    s["scene"],
        }


def collate_fn(batch):
    keys_tensor = ["i_mixed", "e_voxel", "t_gt"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys_tensor}
    out["e_sum_np"] = [b["e_sum_np"] for b in batch]
    out["id"]       = [b["id"]       for b in batch]
    out["scene"]    = [b["scene"]    for b in batch]
    return out


# ============================================================
#  Visualization utils
# ============================================================

def tensor_to_numpy_rgb(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

def to_3ch_grayscale(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[1] == 1:
        return tensor.repeat(1, 3, 1, 1)
    return tensor


# ============================================================
#  Main
# ============================================================

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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

    model = EventGuidedSIRR(
        base       = train_args.get("base", 48),
        voxel_bins = train_args.get("voxel_bins", 5),
        ch_cap     = train_args.get("ch_cap", 192),
        norm       = train_args.get("norm", "gn"),
    ).to(device)

    model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
    model.eval()

    if have_lpips:
        lpips_fn = lpips.LPIPS(net="alex", version="0.1").to(device)
        lpips_fn.eval()

    test_ds = RealDVSDataset(args.data_root, padding=args.padding)
    if len(test_ds) == 0:
        print("No samples found. Exiting.")
        return

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    os.makedirs(SAVE_VIS_DIR, exist_ok=True)

    metrics = {"psnr": [], "ssim": [], "lpips": [], "baseline_psnr": []}
    vis_count = 0

    print(f"\nStarting RealDVS Evaluation ...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            i_mixed   = batch["i_mixed"].to(device)
            e_voxel   = batch["e_voxel"].to(device)
            t_gt      = batch["t_gt"].to(device)
            e_sum_np  = batch["e_sum_np"][0]
            sample_id = batch["id"][0]

            i_mixed_orig = i_mixed
            i_mixed      = to_3ch_grayscale(i_mixed)
            t_gt_3ch     = to_3ch_grayscale(t_gt)
            events       = torch.zeros_like(e_voxel) if args.dummy else e_voxel

            # Reflection Padding
            if args.padding:
                _, _, H, W = i_mixed.shape
                pad_h = (16 - (H % 16)) % 16
                pad_w = (16 - (W % 16)) % 16
                if pad_h > 0 or pad_w > 0:
                    i_mixed = torch.nn.functional.pad(i_mixed, (0, pad_w, 0, pad_h), mode="reflect")
                    events  = torch.nn.functional.pad(events,  (0, pad_w, 0, pad_h), mode="reflect")

                preds = model(i_mixed, events)
                t_est = preds["T_estimate"]

                if pad_h > 0 or pad_w > 0:
                    t_est        = t_est[:, :, :H, :W]
                    i_mixed_orig = i_mixed_orig[:, :, :H, :W]
            else:
                preds = model(i_mixed, events)
                t_est = preds["T_estimate"]

            t_est_gray = t_est.mean(dim=1, keepdim=True)
            t_est_3ch  = t_est_gray.repeat(1, 3, 1, 1)

            # Compute classical metrics on CPU numpy
            gt_t_float  = tensor_to_numpy_rgb(t_gt_3ch[0]).astype(np.float32)  / 255.0
            est_t_float = tensor_to_numpy_rgb(t_est_3ch[0]).astype(np.float32) / 255.0
            mixed_float = tensor_to_numpy_rgb(to_3ch_grayscale(i_mixed_orig)[0]).astype(np.float32) / 255.0

            p_t    = compute_psnr(gt_t_float, est_t_float, data_range=1.0)
            s_t    = compute_ssim(gt_t_float, est_t_float, channel_axis=-1, data_range=1.0)
            p_base = compute_psnr(gt_t_float, mixed_float, data_range=1.0)

            # Direct GPU LPIPS calculation
            if have_lpips:
                lpips_val = lpips_fn((t_est_3ch * 2.0 - 1.0), (t_gt_3ch * 2.0 - 1.0)).item()
            else:
                lpips_val = 0.0

            metrics["psnr"].append(p_t)
            metrics["ssim"].append(s_t)
            metrics["lpips"].append(lpips_val)
            metrics["baseline_psnr"].append(p_base)

            # Visualization
            if vis_count < args.num_vis:
                mixed_np = tensor_to_numpy_rgb(to_3ch_grayscale(i_mixed_orig)[0])
                gt_t_np  = tensor_to_numpy_rgb(t_gt_3ch[0])
                est_t_np = tensor_to_numpy_rgb(t_est_3ch[0])
                
                H_crop, W_crop = mixed_np.shape[:2]
                e_sum_vis = cv2.resize(e_sum_np, (W_crop, H_crop), interpolation=cv2.INTER_AREA) if e_sum_np.shape[:2] != (H_crop, W_crop) else e_sum_np

                grid = np.concatenate([mixed_np, e_sum_vis, est_t_np, gt_t_np], axis=1)
                PILImage.fromarray(grid).save(os.path.join(SAVE_VIS_DIR, f"{sample_id}_psnrT{p_t:.1f}.png"))
                vis_count += 1

    print("\n" + "=" * 50)
    print("  OVERALL RESULTS")
    print("=" * 50)
    
    avg_base = np.mean(metrics["baseline_psnr"])
    avg_psnr = np.mean(metrics["psnr"])
    avg_ssim = np.mean(metrics["ssim"])
    avg_lpips = np.mean(metrics["lpips"])
    improvement = avg_psnr - avg_base

    print(f"  Samples tested      : {len(metrics['psnr'])}")
    print(f"  Baseline PSNR       : {avg_base:.2f} dB")
    print(f"  Model PSNR (T_est)  : {avg_psnr:.2f} dB ({'+' if improvement >= 0 else ''}{improvement:.2f} dB)")
    print(f"  Model SSIM (T_est)  : {avg_ssim:.4f}")
    if have_lpips:
        print(f"  Model LPIPS (T_est) : {avg_lpips:.4f}")
    print("=" * 50)
    print(f"\nVisuals saved in: {SAVE_VIS_DIR}/\n")

if __name__ == "__main__":
    main()



"""
python test_RealDVS.py \
  --data_root /medias/db/ImagingSecurity_misc/melcarne/E-SIRR/utils/ER-100_RealDVS_Dataset/processed \
  --checkpoint /medias/db/ImagingSecurity_misc/melcarne/Events_SIRR/outputs/train/coco_coco_blend_05_cross_stage_gray/C48_cap192_e2e_lr0.0001_bs4_wG_0.0_wExcl_0.02_conf_map_cross_stage_gray/checkpoints/best_model.pth \
  --device cuda:0 
"""