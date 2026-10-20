"""
Dataset and DataLoader for Event-Guided Single Image Reflection Removal (SIRR)
==============================================================================

Dataset structure on disk (one folder per sample, produced by generate_dataset.py):

    <dataset_root>/<sample_id>/
        <sample_id>_T_gt.jpg      — clean background  alpha*T          [0,255] uint8 BGR
        <sample_id>_I_mixed.jpg   — mixed input       alpha*T + beta*R [0,255] uint8 BGR
        <sample_id>_R_warped.jpg  — reflection layer  beta*R_warped    [0,255] uint8 BGR
        <sample_id>_E_voxel.npy   — temporal voxel    (5,H,W) float32 normalised in [-1,1]
        <sample_id>_meta.txt      — generation metadata (alpha, beta, paths, ...)

Each __getitem__ returns a dict:
    i_mixed      : (3, H, W) float32 in [0, 1]
    t_gt         : (3, H, W) float32 in [0, 1]
    r_warped     : (3, H, W) float32 in [0, 1]
    e_voxel      : (5, H, W) float32 in [-1, 1]
    sample_id    : str

Data augmentation (training only):
    - Random horizontal flip applied consistently to all four spatial tensors.
    - Sparse voxel augmentation: randomly drops voxel pixels to simulate the
      low-density distribution observed on SIR2+ (avg density ~0.009).
      Applied only during training (augment=True) so validation is unaffected.
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import Optional
from tqdm import tqdm


# ============================================================
# Helpers
# ============================================================

def rgb_to_gray(x):
    """Convert (3,H,W) tensor [0,1] to (H,W) grayscale."""
    return 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]


def compute_event_correlation(dataset, max_samples=500):
    corr_T_list = []
    corr_R_list = []
    ratio_list  = []

    for i in tqdm(range(min(len(dataset), max_samples))):
        sample = dataset[i]

        T = sample["t_gt"]       # (3,H,W)
        R = sample["r_warped"]   # (3,H,W)
        E = sample["e_voxel"]    # (5,H,W)

        e_sum  = torch.max(torch.abs(E), dim=0)[0]  # (H,W)
        T_gray = rgb_to_gray(T)
        R_gray = rgb_to_gray(R)

        e_flat = e_sum.flatten().numpy()
        T_flat = T_gray.flatten().numpy()
        R_flat = R_gray.flatten().numpy()

        if np.std(e_flat) < 1e-6:
            continue  # skip degenerate (all-zero voxel)

        corr_T = np.corrcoef(e_flat, T_flat)[0, 1]
        corr_R = np.corrcoef(e_flat, R_flat)[0, 1]
        ratio  = corr_R / (corr_T + 1e-6)

        corr_T_list.append(corr_T)
        corr_R_list.append(corr_R)
        ratio_list.append(ratio)

    print("\n===== EVENT CORRELATION RESULTS =====")
    print(f"Samples used : {len(ratio_list)}")
    print(f"Mean corr_T  : {np.mean(corr_T_list):.4f}")
    print(f"Mean corr_R  : {np.mean(corr_R_list):.4f}")
    print(f"Mean ratio   : {np.mean(ratio_list):.4f}")
    print(f"Median ratio : {np.median(ratio_list):.4f}")
    print("\nRatio distribution:")
    print(f"  >1 : {(np.array(ratio_list) > 1).mean()*100:.1f}%")
    print(f"  <1 : {(np.array(ratio_list) < 1).mean()*100:.1f}%")

    return corr_T_list, corr_R_list, ratio_list


def _load_rgb(path: str) -> np.ndarray:
    """
    Load a BGR image with OpenCV and return RGB float32 in [0, 1].
    Shape: (H, W, 3).
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _load_voxel(path: str) -> np.ndarray:
    """
    Load a pre-computed event voxel grid from .npy.
    Expected shape: (num_bins, H, W) = (5, 320, 320), float32 in [-1, 1].
    """
    voxel = np.load(path).astype(np.float32)
    if voxel.ndim != 3:
        raise ValueError(f"Expected 3D voxel (bins,H,W), got shape {voxel.shape}")
    return voxel


def _hwc_to_chw(x: np.ndarray) -> torch.Tensor:
    """(H, W, C) numpy float32 → (C, H, W) float32 tensor."""
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


# ============================================================
# Dataset
# ============================================================

class SIRRDataset(Dataset):
    """
    PyTorch Dataset for Event-Guided Single Image Reflection Removal.

    Args:
        dataset_root       : path to the DVS-Voltmeter output root
        sample_ids         : explicit list of sample_id strings (None = scan all)
        augment            : enable training augmentations (flip + sparse voxel)
        flip_prob          : probability of random horizontal flip (default 0.5)
        gray_mode          : convert RGB images to grayscale pseudo-RGB
        sparse_voxel_prob  : probability of applying sparse voxel augmentation
                             per sample (default 0.80). Only active when augment=True.
        sparse_keep_range  : (min, max) fraction of voxel pixels to keep when
                             sparse augmentation fires (default (0.005, 0.05)).
                             Chosen to match SIR2+ observed avg density ~0.009.
    """

    REQUIRED_SUFFIXES = ("_T_gt.jpg", "_I_mixed.jpg", "_R_warped.jpg", "_E_voxel.npy")

    def __init__(self,
                 dataset_root:      str,
                 sample_ids:        Optional[list] = None,
                 augment:           bool  = False,
                 flip_prob:         float = 0.5,
                 gray_mode:         bool  = False,
                 sparse_voxel_prob: float = 0.80,
                 sparse_keep_range: tuple = (0.005, 0.05)):
        super().__init__()
        self.root              = Path(dataset_root)
        self.augment           = augment
        self.flip_prob         = flip_prob
        self.gray_mode         = gray_mode
        self.sparse_voxel_prob = sparse_voxel_prob
        self.sparse_keep_range = sparse_keep_range

        # --- Discover valid samples ---
        candidates = sample_ids if sample_ids is not None else sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )

        self.samples = []
        n_skipped = 0
        for sid in candidates:
            folder = self.root / sid
            if all((folder / (sid + suf)).exists() for suf in self.REQUIRED_SUFFIXES):
                self.samples.append(sid)
            else:
                n_skipped += 1

        if n_skipped > 0:
            print(f"[SIRRDataset] WARNING: {n_skipped} sample(s) skipped "
                  f"(missing files) in {dataset_root}")
        print(f"[SIRRDataset] {len(self.samples)} valid samples "
              f"({'train+aug' if augment else 'val/test'}) "
              f"from {dataset_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _sparse_augment_voxel(self, e_voxel: torch.Tensor) -> torch.Tensor:
        """
        Simulate the low-density voxel distribution observed on real data (SIR2+)
        without regenerating the dataset.

        Motivation:
            The COCO-COCO training set contains dense voxels (strong reflections,
            fast motion). Real scenes (SIR2+) have avg voxel density ~0.009,
            i.e. fewer than 1% of pixels are active. This mismatch causes the
            EventConfidenceMap to never learn the RGB-fallback behaviour needed
            for sparse inputs, degrading performance on real data.

        Implementation:
            With probability sparse_voxel_prob, a random binary mask is applied
            to the voxel tensor, keeping only keep_prob ∈ sparse_keep_range of
            pixels. The rest are zeroed out. This forces the confidence map to
            produce C≈0 on sparse inputs during training.

        Args:
            e_voxel: (bins, H, W) float32 tensor in [-1, 1] — already on CPU.

        Returns:
            e_voxel (possibly masked), same shape and dtype.
        """
        if random.random() < self.sparse_voxel_prob:
            keep_prob = random.uniform(*self.sparse_keep_range)
            # Bernoulli mask: each pixel independently kept with prob keep_prob
            mask = (torch.rand_like(e_voxel) < keep_prob).float()
            return e_voxel * mask
        return e_voxel

    def __getitem__(self, idx: int) -> dict:
        sid    = self.samples[idx]
        folder = self.root / sid

        # --- Load all modalities from disk ---
        i_mixed  = _load_rgb(str(folder / f"{sid}_I_mixed.jpg"))    # (H,W,3) [0,1]
        t_gt     = _load_rgb(str(folder / f"{sid}_T_gt.jpg"))       # (H,W,3) [0,1]
        r_warped = _load_rgb(str(folder / f"{sid}_R_warped.jpg"))   # (H,W,3) [0,1]
        e_voxel  = _load_voxel(str(folder / f"{sid}_E_voxel.npy"))  # (5,H,W) [-1,1]

        # --- Convert to tensors ---
        t_i_mixed  = _hwc_to_chw(i_mixed)   # (3,H,W) float32
        t_t_gt     = _hwc_to_chw(t_gt)      # (3,H,W) float32
        t_r_warped = _hwc_to_chw(r_warped)  # (3,H,W) float32
        t_e_voxel  = torch.from_numpy(e_voxel)  # (5,H,W) float32

        # --- Training augmentations ---
        if self.augment:
            # 1. Random horizontal flip (consistent across all modalities)
            #    Flip is physically valid: a mirrored scene with mirrored events
            #    and mirrored reflection is still a valid SIRR sample.
            if random.random() < self.flip_prob:
                t_i_mixed  = TF.hflip(t_i_mixed)
                t_t_gt     = TF.hflip(t_t_gt)
                t_r_warped = TF.hflip(t_r_warped)
                # Voxel: (5,H,W) — flip along W dimension (dim=2)
                t_e_voxel  = torch.flip(t_e_voxel, dims=[2])

            # 2. Sparse voxel augmentation — simulates SIR2+ density distribution.
            #    Applied AFTER tensor conversion so torch.rand_like works correctly.
            #    Applied AFTER flip so the mask is not accidentally un-flipped.
            t_e_voxel = self._sparse_augment_voxel(t_e_voxel)

        # --- Optional grayscale mode (pseudo-RGB: 3 identical channels) ---
        if self.gray_mode:
            t_i_mixed  = TF.rgb_to_grayscale(t_i_mixed,  num_output_channels=3)
            t_t_gt     = TF.rgb_to_grayscale(t_t_gt,     num_output_channels=3)
            t_r_warped = TF.rgb_to_grayscale(t_r_warped, num_output_channels=3)

        return {
            "i_mixed":   t_i_mixed,    # (3,H,W) float32 [0,1]
            "t_gt":      t_t_gt,       # (3,H,W) float32 [0,1]
            "r_warped":  t_r_warped,   # (3,H,W) float32 [0,1]
            "e_voxel":   t_e_voxel,    # (5,H,W) float32 [-1,1]
            "sample_id": sid,
        }


# ============================================================
# Train / Val split
# ============================================================

def make_train_val_split(dataset_root: str,
                         val_fraction: float = 0.20,
                         seed:         int   = 42) -> tuple:
    """
    Deterministic random train/val split on all valid sample folders.

    Returns:
        (train_ids, val_ids): two sorted lists of sample_id strings.
    """
    root = Path(dataset_root)

    all_ids = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and all(
            (d / (d.name + suf)).exists()
            for suf in SIRRDataset.REQUIRED_SUFFIXES
        )
    )

    if len(all_ids) == 0:
        raise RuntimeError(f"No valid samples found in {dataset_root}")

    rng       = np.random.default_rng(seed)
    indices   = rng.permutation(len(all_ids))
    n_val     = max(1, int(len(all_ids) * val_fraction))
    val_idx   = set(indices[:n_val].tolist())
    train_idx = set(indices[n_val:].tolist())

    train_ids = [all_ids[i] for i in sorted(train_idx)]
    val_ids   = [all_ids[i] for i in sorted(val_idx)]

    print(f"[split] Total valid: {len(all_ids)} | "
          f"Train: {len(train_ids)} | Val: {len(val_ids)} "
          f"(seed={seed}, val_frac={val_fraction:.0%})")
    return train_ids, val_ids


# ============================================================
# DataLoader factory
# ============================================================

def build_dataloaders(dataset_root:      str,
                      batch_size:        int   = 4,
                      num_workers:       int   = 4,
                      val_fraction:      float = 0.20,
                      seed:              int   = 42,
                      flip_prob:         float = 0.5,
                      pin_memory:        bool  = True,
                      gray_mode:         bool  = False,
                      sparse_voxel_prob: float = 0.80,
                      sparse_keep_range: tuple = (0.005, 0.05)) -> tuple:
    """
    Build train and validation DataLoaders from a single dataset root.

    Args:
        sparse_voxel_prob  : passed to SIRRDataset for training set only.
        sparse_keep_range  : passed to SIRRDataset for training set only.

    Returns:
        (train_loader, val_loader, train_dataset, val_dataset)
    """
    train_ids, val_ids = make_train_val_split(dataset_root, val_fraction, seed)

    train_ds = SIRRDataset(
        dataset_root,
        sample_ids        = train_ids,
        augment           = True,
        flip_prob         = flip_prob,
        gray_mode         = gray_mode,
        sparse_voxel_prob = sparse_voxel_prob,
        sparse_keep_range = sparse_keep_range,
    )
    val_ds = SIRRDataset(
        dataset_root,
        sample_ids = val_ids,
        augment    = False,   # no augmentation on validation
        gray_mode  = gray_mode,
        # sparse_voxel_prob / sparse_keep_range intentionally not passed:
        # validation always sees the original dense voxel from disk.
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        drop_last          = True,
        generator          = g,
        persistent_workers = (num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        drop_last          = False,
        persistent_workers = (num_workers > 0),
    )

    return train_loader, val_loader, train_ds, val_ds


# ============================================================
# Sanity check
# ============================================================

if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DATASET_ROOT = "/medias/db/ImagingSecurity_misc/SIRR_Datasets/COCO_COCO_EVENTS_A_BLENDED_TRAIN_30_05/dvs"

    train_loader, val_loader, train_ds, val_ds = build_dataloaders(
        DATASET_ROOT,
        batch_size        = 8,
        num_workers       = 0,
        val_fraction      = 0.20,
        seed              = 42,
        gray_mode         = True,
        sparse_voxel_prob = 0.80,
        sparse_keep_range = (0.005, 0.05),
    )

    compute_event_correlation(train_ds, max_samples=500)

    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")

    batch = next(iter(train_loader))
    print("\nFirst train batch:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:12s} shape={tuple(v.shape)}  "
                  f"dtype={v.dtype}  "
                  f"min={v.min():.3f}  max={v.max():.3f}")
        else:
            print(f"  {k:12s} = {v}")

    B = batch["i_mixed"].shape[0]
    assert batch["i_mixed"].shape  == (B, 3, 320, 320)
    assert batch["t_gt"].shape     == (B, 3, 320, 320)
    assert batch["r_warped"].shape == (B, 3, 320, 320)
    assert batch["e_voxel"].shape  == (B, 5, 320, 320)
    assert batch["i_mixed"].min()  >= 0.0
    assert batch["i_mixed"].max()  <= 1.0
    print("\nAll shape/range assertions passed.")

    # --- Visualise one sample ---
    sample = {k: v[0] for k, v in batch.items() if isinstance(v, torch.Tensor)}

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    titles = ["I_mixed", "T_gt", "R_warped", "E_voxel sum"]

    for ax, key, title in zip(axes[:3], ["i_mixed", "t_gt", "r_warped"], titles[:3]):
        img = sample[key].permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title)
        ax.axis("off")

    vox     = sample["e_voxel"].numpy()
    max_pos = np.max(np.maximum( vox, 0), axis=0)
    max_neg = np.max(np.maximum(-vox, 0), axis=0)
    vox_rgb = np.full((vox.shape[1], vox.shape[2], 3), [0.109, 0.010, 0.2], dtype=np.float32)
    is_pos  = (max_pos > 0.05) & (max_pos >= max_neg)
    is_neg  = (max_neg > 0.05) & (max_neg >  max_pos)
    vox_rgb[is_pos] = [0.874, 0.451, 1.0]
    vox_rgb[is_neg] = [0.863, 0.863, 0.863]

    axes[3].imshow(vox_rgb)
    axes[3].set_title(titles[3])
    axes[3].axis("off")

    out_path = "/medias/db/ImagingSecurity_misc/melcarne/Events_SIRR/dataset/sirr_dataset_sample_gray.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nVisualization saved to {out_path}")
    plt.close()

    val_batch = next(iter(val_loader))
    print(f"\nFirst val batch: i_mixed shape = {tuple(val_batch['i_mixed'].shape)}")
    print("Dataset sanity check complete.")