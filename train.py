"""
Training script for Event-Guided Single Image Reflection Removal (SIRR)
========================================================================

Trains the full model end-to-end (RFB event head + confidence-modulated
EFF fusion + cross-stage skip connections — see model/ESIRR_Net.py).

Usage — single GPU, fresh start:
    python -u train.py --device 0 --base 48 --ch_cap 192 | tee train.log

Usage — resume from checkpoint:
    python -u train.py --device 0 --base 48 --ch_cap 192 \
        --resume ./outputs/train/C48_cap192_lr0.0001_bs4/checkpoints/best_model.pth \
        | tee train_resumed.log

Usage — multi-GPU (DDP):
    torchrun --nproc_per_node=2 train.py --ddp --base 48 --ch_cap 192 --epochs 200 \
        | tee train_48_cap192.log

Resume behaviour
----------------
When --resume is provided:
  - model weights, EMA weights, optimizer state, scheduler state are all restored
  - training resumes from epoch (saved_epoch + 1)
  - best_val_loss and best_psnr are restored so best-model logic works correctly
  - the learning rate follows the restored scheduler state, unless --resume_lr
    is explicitly passed to override it (e.g. for a lower-LR fine-tuning phase)
  - loss history is reloaded from the CSV in save_dir (if present) so that
    the final loss curves cover the full training run, not just the resumed part
"""

import os
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from PIL import Image as PILImage

from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity  as compute_ssim

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from model.ESIRR_Net    import EventGuidedSIRR, SIRRLoss
from dataset.ESIRR_dataset import SIRRDataset, make_train_val_split


# ============================================================
#  Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train Event-Guided SIRR (full model)")

    # Hardware
    p.add_argument("--device",       type=int,   default=0)
    p.add_argument("--ddp",          action="store_true")
    p.add_argument("--num_workers",  type=int,   default=4)

    # Dataset
    p.add_argument("--dataset_root", type=str,
                   default="./data/synthetic")
    p.add_argument("--val_fraction", type=float, default=0.20)
    p.add_argument("--seed",         type=int,   default=42)

    # Training
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Resume
    p.add_argument("--resume",        type=str,   default=None,
                   help="Path to checkpoint (.pth) to resume training from. "
                        "Restores model, EMA, optimizer, scheduler and epoch.")
    p.add_argument("--resume_lr",     type=float, default=None,
                   help="If set, overrides the learning rate after resuming "
                        "(e.g. for a lower-LR fine-tuning phase). By default "
                        "the LR follows the restored scheduler state.")

    # Model
    p.add_argument("--base",         type=int,   default=48)
    p.add_argument("--ch_cap",       type=int,   default=192,
                   help="Channel cap (e.g. 192 for base=48). None = uncapped.")
    p.add_argument("--voxel_bins",   type=int,   default=5)
    p.add_argument("--dropout",      type=float, default=0.0)

    # Loss weights
    p.add_argument("--w_l1_R",   type=float, default=1.0)
    p.add_argument("--w_perc_R", type=float, default=0.1)
    p.add_argument("--w_l1_T",   type=float, default=1.0)
    p.add_argument("--w_perc_T", type=float, default=0.1)
    p.add_argument("--w_excl",   type=float, default=0.02)

    # EMA
    p.add_argument("--ema_decay",    type=float, default=0.999)

    # Logging
    p.add_argument("--vis_every",    type=int,   default=20)
    p.add_argument("--output_dir",   type=str,
                   default=str(project_root / "outputs" / "train"))

    # Data / augmentation
    p.add_argument("--gray",         action="store_true",
                   help="Grayscale images (Pseudo-RGB) — use for grayscale "
                        "real-world data such as ER-100.")

    p.add_argument("--sparse_voxel_prob", type=float, default=0.0,
                help="Probability of sparse voxel augmentation per sample "
                        "(0.0 = disabled, 0.8 = recommended for SIR2+ fine-tuning)")
    p.add_argument("--sparse_keep_min",  type=float, default=0.005,
                help="Min fraction of voxel pixels to keep when sparse aug fires")
    p.add_argument("--sparse_keep_max",  type=float, default=0.05,
                help="Max fraction of voxel pixels to keep when sparse aug fires")

    return p.parse_args()


# ============================================================
#  Run Naming Convention (Uniformed for Train & Test)
# ============================================================

def build_run_name(args):
    """
    Constructs a standardized run folder name based on training hyperparameters.
    This exact logic should be mirrored (or imported) in evaluation scripts
    to easily reconstruct the checkpoint path.
    """
    cap_str = f"cap{args.ch_cap}" if args.ch_cap is not None else "uncapped"
    name = f"C{args.base}_{cap_str}_lr{args.lr}_bs{args.batch_size}"
    if args.w_excl != 0.02:
        name += f"_wExcl_{args.w_excl}"
    if args.gray:
        name += "_gray"
    return name


# ============================================================
#  DDP / device setup
# ============================================================

def setup_ddp(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    def is_main(): return dist.get_rank() == 0
    return device, local_rank, world_size, is_main


def setup_single(args):
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    def is_main(): return True
    return device, args.device, 1, is_main


# ============================================================
#  EMA
# ============================================================

def init_ema(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def update_ema(model: nn.Module, ema_dict: dict, decay: float = 0.999):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                ema_dict[k].mul_(decay).add_(v.detach().cpu(), alpha=1.0 - decay)
            else:
                ema_dict[k].copy_(v.detach().cpu())


def load_ema(model: nn.Module, ema_dict: dict) -> dict:
    """Load EMA weights for validation; returns backup of current weights."""
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(
        {k: v.to(next(model.parameters()).device) for k, v in ema_dict.items()},
        strict=True)
    return backup


# ============================================================
#  Resume helper
# ============================================================

def load_checkpoint(path: str, model: nn.Module, ema: dict,
                    optimizer, scheduler, device) -> dict:
    """
    Loads a checkpoint saved by this script and restores all stateful objects.
    """
    ckpt = torch.load(path, map_location=device)

    # Restore model weights
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Restore EMA weights (update the existing ema dict in-place)
    if "ema_state_dict" in ckpt:
        for k, v in ckpt["ema_state_dict"].items():
            if k in ema:
                ema[k].copy_(v.cpu())

    # Restore optimizer and scheduler
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch   = ckpt.get("epoch", 0) + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    best_psnr     = ckpt.get("best_psnr",     0.0)

    return {
        "start_epoch":   start_epoch,
        "best_val_loss": best_val_loss,
        "best_psnr":     best_psnr,
    }


def reload_history(csv_path: str) -> defaultdict:
    """
    Reloads loss history from a previously saved CSV.
    """
    history = defaultdict(list)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for col in df.columns:
            if col != "epoch":
                history[col] = df[col].tolist()
    return history


# ============================================================
#  Visualisation
# ============================================================

def tensor_to_numpy_rgb(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def voxel_to_rgb_numpy(voxel: torch.Tensor) -> np.ndarray:
    v   = voxel.detach().cpu().numpy()
    pos = np.max(np.maximum( v, 0), axis=0)
    neg = np.max(np.maximum(-v, 0), axis=0)
    out = np.zeros((v.shape[1], v.shape[2], 3), dtype=np.uint8)
    thr = 0.05
    out[(pos > thr) & (pos >= neg)] = [255,   0,   0]
    out[(neg > thr) & (neg >  pos)] = [  0, 255,   0]
    return out


def save_vis_grid(batch: dict, preds: dict, vis_dir: str, epoch: int,
                  sample_idx: int = 0) -> float:
    os.makedirs(vis_dir, exist_ok=True)
    i = sample_idx

    i_mixed  = tensor_to_numpy_rgb(batch["i_mixed"][i])
    t_gt     = tensor_to_numpy_rgb(batch["t_gt"][i])
    r_warped = tensor_to_numpy_rgb(batch["r_warped"][i])
    e_voxel  = voxel_to_rgb_numpy( batch["e_voxel"][i])
    t_est    = tensor_to_numpy_rgb(preds["T_estimate"][i])
    r_est    = tensor_to_numpy_rgb(preds["R_estimate"][i])

    residual = np.abs(i_mixed.astype(np.float32) - t_est.astype(np.float32))
    residual = np.clip(residual * 3.0, 0, 255).astype(np.uint8)

    H, W = i_mixed.shape[:2]
    pad, cols = 4, 4
    grid = np.zeros((2*H + 3*pad, cols*W + (cols+1)*pad, 3), dtype=np.uint8)

    def place(row, col, img):
        y0 = pad + row*(H+pad); x0 = pad + col*(W+pad)
        grid[y0:y0+H, x0:x0+W] = img

    place(0,0,i_mixed); place(0,1,t_gt);    place(0,2,r_warped); place(0,3,e_voxel)
    place(1,0,t_est);   place(1,1,residual); place(1,2,r_est)

    PILImage.fromarray(grid).save(
        os.path.join(vis_dir, f"epoch{epoch:04d}_grid.png"))
    PILImage.fromarray(t_est).save(
        os.path.join(vis_dir, f"epoch{epoch:04d}_T_est.png"))
    PILImage.fromarray(r_est).save(
        os.path.join(vis_dir, f"epoch{epoch:04d}_R_est.png"))
    PILImage.fromarray(t_gt).save(
        os.path.join(vis_dir, f"epoch{epoch:04d}_T_gt.png"))
    PILImage.fromarray(i_mixed).save(
        os.path.join(vis_dir, f"epoch{epoch:04d}_I_mixed.png"))

    return compute_psnr(t_gt.astype(np.float32)/255.0,
                        t_est.astype(np.float32)/255.0,
                        data_range=1.0)


# ============================================================
#  Metrics
# ============================================================

def batch_metrics(t_est: torch.Tensor,
                  t_gt:  torch.Tensor) -> tuple:
    psnr_vals, ssim_vals = [], []
    est_np = t_est.detach().cpu().numpy().transpose(0,2,3,1)
    gt_np  = t_gt.detach().cpu().numpy().transpose(0,2,3,1)
    for p, g in zip(est_np, gt_np):
        p = np.clip(p, 0.0, 1.0); g = np.clip(g, 0.0, 1.0)
        psnr_vals.append(compute_psnr(g, p, data_range=1.0))
        ssim_vals.append(compute_ssim(g, p, channel_axis=-1, data_range=1.0))
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


# ============================================================
#  Main
# ============================================================

def main():
    args = parse_args()

    if args.ddp:
        device, local_rank, world_size, is_main = setup_ddp(args)
    else:
        device, local_rank, world_size, is_main = setup_single(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Output dirs ---
    run_tag  = build_run_name(args)
    save_dir = os.path.join(args.output_dir, run_tag)
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    vis_dir  = os.path.join(save_dir, "visualisations")
    csv_path = os.path.join(save_dir, "loss_curves.csv")

    if is_main():
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(vis_dir,  exist_ok=True)

    # --- Dataset ---
    train_ids, val_ids = make_train_val_split(
        args.dataset_root, args.val_fraction, args.seed)

    train_ds = SIRRDataset(args.dataset_root, sample_ids=train_ids,
                       augment=True, flip_prob=0.5, gray_mode=args.gray,
                       sparse_voxel_prob=args.sparse_voxel_prob,
                       sparse_keep_range=(args.sparse_keep_min, args.sparse_keep_max))

    val_ds   = SIRRDataset(args.dataset_root, sample_ids=val_ids,
                           augment=False, gray_mode=args.gray)

    if args.ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False)
        train_loader  = DataLoader(train_ds, batch_size=args.batch_size,
                                   sampler=train_sampler, num_workers=args.num_workers,
                                   pin_memory=True, drop_last=True,
                                   persistent_workers=True)
        val_loader    = DataLoader(val_ds,   batch_size=args.batch_size,
                                   sampler=val_sampler,   num_workers=args.num_workers,
                                   pin_memory=True, drop_last=False,
                                   persistent_workers=True)
    else:
        g = torch.Generator(); g.manual_seed(args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True,
                                  generator=g, persistent_workers=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=False,
                                  persistent_workers=True)

    # --- Model (full model — see model/ESIRR_Net.py) ---
    model = EventGuidedSIRR(
        base=args.base, voxel_bins=args.voxel_bins,
        ch_cap=args.ch_cap, norm="gn", dropout=args.dropout,
    ).to(device)

    if args.ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    def unwrap(m): return m.module if args.ddp else m

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- Loss ---
    criterion = SIRRLoss(
            w_l1_R=args.w_l1_R,
            w_perc_R=args.w_perc_R,
            w_l1_T=args.w_l1_T,
            w_perc_T=args.w_perc_T,
            w_excl=args.w_excl,
        ).to(device)

    # --- EMA ---
    ema = init_ema(unwrap(model))

    # --- Training state defaults ---
    start_epoch   = 1
    best_val_loss = float("inf")
    best_psnr     = 0.0
    history       = defaultdict(list)

    # ------------------------------------------------------------------ #
    #  RESUME
    # ------------------------------------------------------------------ #
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

        if is_main():
            print(f"\n[Resume] Loading checkpoint: {args.resume}")

        ckpt_info = load_checkpoint(
            args.resume, unwrap(model), ema, optimizer, scheduler, device)

        if args.resume_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.resume_lr
            if is_main():
                print(f"[Resume] Learning rate overridden to {args.resume_lr:.2e} "
                      f"(--resume_lr)")

        start_epoch   = ckpt_info["start_epoch"]
        best_val_loss = ckpt_info["best_val_loss"]
        best_psnr     = ckpt_info["best_psnr"]

        history = reload_history(csv_path)

        if is_main():
            print(f"[Resume] Resuming from epoch {start_epoch}  "
                  f"(best_val_loss={best_val_loss:.4f}, "
                  f"best_psnr={best_psnr:.2f} dB)")

    # ------------------------------------------------------------------ #
    #  Print config (after resume so start_epoch is correct)
    # ------------------------------------------------------------------ #
    if is_main():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ch_cap_display = (args.ch_cap if args.ch_cap
                          else f"{args.base}*8={args.base*8} (uncapped)")
        resume_str = f"  Resume from   : {args.resume}\n" if args.resume else ""
        print(f"""
=== Event-Guided SIRR Training ===
  Device        : {device}
  World size    : {world_size}
  Dataset root  : {args.dataset_root}
  Batch size    : {args.batch_size}
  Epochs        : {start_epoch} → {args.epochs}
  LR            : {args.lr}  (current: {optimizer.param_groups[0]['lr']:.2e})
  Base channels : {args.base}  |  ch_cap : {ch_cap_display}
  Model params  : {n_params/1e6:.2f} M
{resume_str}  Train samples : {len(train_ds)} ({len(train_loader)} batches)
  Val samples   : {len(val_ds)} ({len(val_loader)} batches)
  Save dir      : {save_dir}
""")

    if start_epoch > args.epochs:
        if is_main():
            print(f"[Resume] start_epoch={start_epoch} >= epochs={args.epochs}. "
                  f"Nothing to do. Increase --epochs.")
        return

    # ============================================================
    #  Training loop
    # ============================================================
    if is_main():
        print("=== Starting Training ===\n")

    for epoch in range(start_epoch, args.epochs + 1):

        if args.ddp:
            train_sampler.set_epoch(epoch)

        # ------------------------------------------------------------------ #
        #  TRAIN
        # ------------------------------------------------------------------ #
        model.train()
        running = defaultdict(float)
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:03d}/{args.epochs} [Train]",
                    disable=not is_main())

        for batch in pbar:
            i_mixed = batch["i_mixed"].to(device, non_blocking=True)
            e_voxel = batch["e_voxel"].to(device, non_blocking=True)
            t_gt    = batch["t_gt"].to(device,    non_blocking=True)
            r_gt    = batch["r_warped"].to(device, non_blocking=True)

            optimizer.zero_grad()

            preds = model(i_mixed, e_voxel)
            loss_dict = criterion(preds, {"R_warped_gt": r_gt, "T_gt": t_gt})
            loss_dict["total"].backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()
            update_ema(unwrap(model), ema, args.ema_decay)

            for k, v in loss_dict.items():
                running[k] += v.item()

            if is_main():
                pbar.set_postfix({
                    "loss": f"{loss_dict['total'].item():.4f}",
                    "L_s1": f"{loss_dict['L_s1'].item():.3f}",
                    "L_s2": f"{loss_dict['L_s2'].item():.3f}",
                })

        n_train = len(train_loader)
        for k in running:
            history[f"train_{k}"].append(running[k] / n_train)

        # ------------------------------------------------------------------ #
        #  VALIDATION  (EMA weights)
        # ------------------------------------------------------------------ #
        model.eval()
        backup = load_ema(unwrap(model), ema)

        val_running = defaultdict(float)
        psnr_list, ssim_list, psnr_baseline_list = [], [], []
        first_val_batch, first_val_preds = None, None

        with torch.no_grad():
            for b_idx, batch in enumerate(
                    tqdm(val_loader,
                         desc=f"Epoch {epoch:03d}/{args.epochs} [Val  ]",
                         disable=not is_main())):

                i_mixed = batch["i_mixed"].to(device, non_blocking=True)
                e_voxel = batch["e_voxel"].to(device, non_blocking=True)
                t_gt    = batch["t_gt"].to(device,    non_blocking=True)
                r_gt    = batch["r_warped"].to(device, non_blocking=True)

                preds = unwrap(model)(i_mixed, e_voxel)
                loss_dict = criterion(preds, {"R_warped_gt": r_gt, "T_gt": t_gt})

                for k, v in loss_dict.items():
                    val_running[k] += v.item()

                p_net, s = batch_metrics(preds["T_estimate"], t_gt)
                psnr_list.append(p_net)
                ssim_list.append(s)

                p_base, _ = batch_metrics(i_mixed, t_gt)
                psnr_baseline_list.append(p_base)

                if b_idx == 0:
                    first_val_batch = {k: v.cpu() for k, v in batch.items()
                                       if isinstance(v, torch.Tensor)}
                    first_val_preds = {k: v.cpu() for k, v in preds.items()}

        if args.ddp:
            agg = torch.tensor([
                val_running["total"],
                np.sum(psnr_list), np.sum(ssim_list),
                np.sum(psnr_baseline_list), len(psnr_list)
            ], device=device)
            dist.all_reduce(agg)
            avg_val_loss  = agg[0].item() / (len(val_loader) * world_size)
            avg_psnr      = agg[1].item() / agg[4].item()
            avg_ssim      = agg[2].item() / agg[4].item()
            avg_psnr_base = agg[3].item() / agg[4].item()
        else:
            avg_val_loss  = val_running["total"] / len(val_loader)
            avg_psnr      = float(np.mean(psnr_list))
            avg_ssim      = float(np.mean(ssim_list))
            avg_psnr_base = float(np.mean(psnr_baseline_list))

        delta_db = avg_psnr - avg_psnr_base

        n_val = len(val_loader)
        for k in val_running:
            history[f"val_{k}"].append(val_running[k] / n_val)
        history["val_psnr"].append(avg_psnr)
        history["val_ssim"].append(avg_ssim)
        history["val_psnr_baseline"].append(avg_psnr_base)
        history["val_delta_db"].append(delta_db)

        scheduler.step()
        unwrap(model).load_state_dict(backup)

        # ------------------------------------------------------------------ #
        #  Logging & saving (main process only)
        # ------------------------------------------------------------------ #
        if is_main():
            delta_sign = "+" if delta_db >= 0 else ""
            print(
                f"[{epoch:03d}/{args.epochs}] "
                f"Train: {history['train_total'][-1]:.4f}  "
                f"Val: {avg_val_loss:.4f}  "
                f"PSNR: {avg_psnr:.2f} dB  "
                f"(baseline: {avg_psnr_base:.2f} dB  "
                f"ΔdB: {delta_sign}{delta_db:.2f})  "
                f"SSIM: {avg_ssim:.4f}  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_psnr     = avg_psnr
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     unwrap(model).state_dict(),
                    "ema_state_dict":       {k: v.cpu() for k, v in ema.items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss":        best_val_loss,
                    "best_psnr":            best_psnr,
                    "args":                 vars(args),
                }, os.path.join(ckpt_dir, "best_model.pth"))
                print(f"  ✓ Best model saved  "
                      f"(val_loss={best_val_loss:.4f}, "
                      f"PSNR={best_psnr:.2f} dB, ΔdB={delta_sign}{delta_db:.2f})")

            if epoch % 50 == 0:
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     unwrap(model).state_dict(),
                    "ema_state_dict":       {k: v.cpu() for k, v in ema.items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss":        best_val_loss,
                    "best_psnr":            best_psnr,
                    "args":                 vars(args),
                }, os.path.join(ckpt_dir, f"epoch{epoch:04d}.pth"))
                print(f"  Checkpoint saved → epoch{epoch:04d}.pth")

            if epoch % args.vis_every == 0 and first_val_batch is not None:
                epoch_vis_dir = os.path.join(vis_dir, f"epoch{epoch:04d}")
                vis_psnr = save_vis_grid(
                    first_val_batch, first_val_preds,
                    epoch_vis_dir, epoch, sample_idx=0)
                print(f"  Vis saved (sample PSNR={vis_psnr:.2f} dB) "
                      f"→ {epoch_vis_dir}")

            epochs_so_far = len(history["train_total"])
            df = pd.DataFrame({
                "epoch": list(range(1, epochs_so_far + 1)),
                **{k: v for k, v in history.items()}
            })
            df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------ #
    #  Final plot
    # ------------------------------------------------------------------ #
    if is_main():
        try:
            import matplotlib.pyplot as plt
            df = pd.read_csv(csv_path)
            fig, axes = plt.subplots(1, 4, figsize=(22, 5))

            axes[0].plot(df["epoch"], df["train_total"], label="Train")
            axes[0].plot(df["epoch"], df["val_total"],   label="Val")
            axes[0].set_title("Total Loss"); axes[0].legend(); axes[0].grid(True)

            axes[1].plot(df["epoch"], df["train_L_s1"], label="Train L_s1")
            axes[1].plot(df["epoch"], df["val_L_s1"],   label="Val L_s1")
            axes[1].plot(df["epoch"], df["train_L_s2"], label="Train L_s2", ls="--")
            axes[1].plot(df["epoch"], df["val_L_s2"],   label="Val L_s2",   ls="--")
            axes[1].set_title("Stage Losses"); axes[1].legend(); axes[1].grid(True)

            axes[2].plot(df["epoch"], df["val_psnr"],          label="Network PSNR")
            axes[2].plot(df["epoch"], df["val_psnr_baseline"], label="Baseline PSNR",
                         ls="--", color="gray")
            axes[2].set_title("Val PSNR (dB)"); axes[2].legend(); axes[2].grid(True)

            axes[3].plot(df["epoch"], df["val_delta_db"], color="tab:orange")
            axes[3].axhline(0, color="black", ls="--", lw=0.8)
            axes[3].set_title("ΔdB = PSNR(T_est) − PSNR(I_mixed)")
            axes[3].grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"[plot] skipped: {e}")

        print(f"\n=== Training Complete ===")
        print(f"  Best val loss : {best_val_loss:.4f}")
        print(f"  Best PSNR     : {best_psnr:.2f} dB")
        print(f"  Checkpoints   : {ckpt_dir}")
        print(f"  Loss curves   : {csv_path}")

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()