"""
Event-Guided SIRR: Synthetic Dataset Generation
===============================================
Generates synthetic multimodal reflection data using COCO images and DVS-Voltmeter.
Features:
- Smooth 2D cubic spline trajectories for dynamic reflections.
- Non-linear screen blending: I = alpha*T + beta*R - (alpha*T)*(beta*R).
- Gaussian blur on reflections to simulate defocus.
- Synchronised event voxel grid generation using DVS-Voltmeter.
- Strict quality filters to discard degenerate samples (low contrast, out-of-bounds).
"""


"""
FOR PEOPLE CLONING THE REPOSITORY: 
PLEASE MAKE SURE TO HAVE INSTALLED THE DVS VOLTMETER AND TO PASS THE CORRECT PATHS INSIDE THIS CODE.
SAME THING FOR THE COCO DATASET, OR ANY OTHER DATASET YOU DECIDE TO USE INSTEAD.
"""

import os
import re
import random
import math
import subprocess
import tempfile
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# ============================================================
#  CONFIGURATION
# ============================================================

# --- Paths ---
COCO_ROOT = "/medias/db/ImagingSecurity_misc/SIRR_Datasets/COCO/train2017" 
OUT_ROOT  = "./data/synthetic/dvs"

DVS_VOLTMETER_PATH      = "/medias/db/ImagingSecurity_misc/melcarne/Event_simulators/DVS-Voltmeter"
DVS_VOLTMETER_CONDA_ENV = "dvs-voltmeter"
DVS_VOLTMETER_CAMERA    = "DVSCustomSIRR"  # Matches real DAVIS346

# --- Generation Parameters ---
N_SAMPLES        = 11000
ALPHA_RANGE      = (0.70, 0.85)
BETA_RANGE       = (0.35, 0.60)
BLUR_SIGMA_RANGE = (1.10, 3.0)

# --- Rendering & Crop ---
TARGET_SIZE  = (320, 320)  # Final (W, H)
NUM_FRAMES   = 300 
FPS          = 30 
TARGET_FRAME = 150         # Midpoint for synchronous voxel extraction
MARGIN       = 120         # Safety buffer for trajectory displacement
MAX_DISP     = 100         # Spline max displacement (px)

# --- Voxel ---
VOXEL_N_FRAMES = 0.5
VOXEL_NUM_BINS = 5

RANDOM_SEED = 42

# --- Quality Filters ---
MIN_R_GRADIENT           = 10.0
MIN_R_NONZERO_FRAC       = 0.10
MIN_TRAJ_SPEED_PX        = 2.0
MIN_R_CROP_FRAC          = 0.05
MIN_MEAN_DIFF            = 1.0
MIN_DIFF_FOR_VOXEL_CHECK = 40.0
MIN_VOXEL_DENSITY        = 0.010

MAX_R_ATTEMPTS    = 10
MAX_TRAJ_ATTEMPTS = 10


# ============================================================
#  IMAGE PREPROCESSING
# ============================================================

def resize_center_crop(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x0, y0 = (new_w - target_w) // 2, (new_h - target_h) // 2
    return resized[y0:y0 + target_h, x0:x0 + target_w]


# ============================================================
#  QUALITY FILTERS
# ============================================================

def check_R_quality(R_raw):
    gray = cv2.cvtColor(R_raw.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    nonzero_mask = gray > 10
    nonzero_frac = nonzero_mask.sum() / gray.size
    
    if nonzero_frac < MIN_R_NONZERO_FRAC:
        return False, f"R sparse: {nonzero_frac:.2f}"

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    mean_grad = grad_mag[nonzero_mask].mean() if nonzero_mask.any() else 0.0
    
    if mean_grad < MIN_R_GRADIENT:
        return False, f"R smooth: {mean_grad:.2f}"

    return True, "ok"

def check_trajectory_speed(trajectory, target_frame, n_frames_window):
    window_int = max(1, int(math.ceil(n_frames_window))) 
    f_start = int(max(0, target_frame - window_int))
    f_end   = int(min(len(trajectory) - 1, target_frame + window_int))
    window  = trajectory[f_start:f_end + 1]

    if len(window) < 2: return False, 0.0, "window too short"

    diffs = np.diff(window, axis=0)
    speeds = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    mean_speed = speeds.mean()

    if mean_speed < MIN_TRAJ_SPEED_PX:
        return False, mean_speed, f"slow: {mean_speed:.2f} px/f"
    return True, mean_speed, "ok"

def check_R_in_crop(trajectory, target_frame, margin, R_raw, width, height):
    dx, dy = trajectory[target_frame]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    R_warped = cv2.warpAffine(R_raw, M, (width, height))
    R_crop = R_warped[margin:height - margin, margin:width - margin]
    R_gray = cv2.cvtColor(R_crop.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    frac_visible = float((R_gray > 10).mean())
    return frac_visible >= MIN_R_CROP_FRAC, frac_visible

def diagnose_frame_diff(frames, target_frame):
    f1 = frames[target_frame - 1].astype(np.float32)
    f2 = frames[target_frame    ].astype(np.float32)
    f3 = frames[target_frame + 1].astype(np.float32)
    
    gray_diff = ((np.abs(f2 - f1) + np.abs(f3 - f2)) / 2.0).mean(axis=2)
    mean_diff = float(gray_diff.mean())
    return mean_diff >= MIN_MEAN_DIFF, mean_diff

def check_case_b(voxel, T_gt, I_mixed):
    T_gray = cv2.cvtColor(T_gt,    cv2.COLOR_BGR2GRAY).astype(np.float32)
    I_gray = cv2.cvtColor(I_mixed, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff_mean = float(np.mean(np.abs(I_gray - T_gray)))

    if diff_mean <= MIN_DIFF_FOR_VOXEL_CHECK:
        return True, "ok (low diff)"

    voxel_density = float(np.mean(np.max(np.abs(voxel), axis=0) > 0.05))
    if voxel_density < MIN_VOXEL_DENSITY:
        return False, f"empty voxel: density={voxel_density:.4f}"
    return True, "ok"


# ============================================================
#  DATASET PREP
# ============================================================

def collect_image_paths(root):
    exts = {".jpg", ".jpeg", ".png"}
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in exts:
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)

def build_sample_list(n_samples, coco_paths, seed=42):
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    n = len(coco_paths)

    t_indices = [rng.randint(0, n - 1) for _ in range(n_samples)]
    r_indices = [(t + rng.randint(1, n - 1)) % n for t in t_indices] # Ensure T != R

    alphas = np_rng.uniform(ALPHA_RANGE[0], ALPHA_RANGE[1], n_samples)
    betas  = np_rng.uniform(BETA_RANGE[0],  BETA_RANGE[1],  n_samples)
    sigmas = np_rng.uniform(BLUR_SIGMA_RANGE[0], BLUR_SIGMA_RANGE[1], n_samples)

    return [{
        "sample_id":  f"coco_coco_{i:06d}",
        "path_T":     coco_paths[t_indices[i]],
        "path_R":     coco_paths[r_indices[i]],
        "alpha":      float(alphas[i]), 
        "beta":       float(betas[i]),  
        "blur_sigma": float(sigmas[i])
    } for i in range(n_samples)]


# ============================================================
#  SPLINE TRAJECTORY
# ============================================================

def generate_spline_trajectory(num_frames, num_keypoints=6):
    max_safe = MARGIN - 10  
    u_keys = np.linspace(0, 1, num_keypoints)
    pts_x, pts_y = [], []
    center_axis = np.random.choice(['x', 'y'])
    center_direction = np.random.choice([-1.0, 1.0])

    for k in range(num_keypoints):
        if abs(u_keys[k] - 0.5) < 0.15:
            extreme_val = np.random.uniform(max_safe * 0.7, max_safe) * center_direction
            other_val = np.random.uniform(-max_safe * 0.4, max_safe * 0.4)
            if center_axis == 'x':
                pts_x.append(extreme_val); pts_y.append(other_val)
            else:
                pts_x.append(other_val); pts_y.append(extreme_val)
            center_direction *= -1.0
        else:
            pts_x.append(np.random.uniform(-max_safe * 0.5, max_safe * 0.5))
            pts_y.append(np.random.uniform(-max_safe * 0.5, max_safe * 0.5))

    pts_x[0], pts_y[0], pts_x[-1], pts_y[-1] = 0.0, 0.0, 0.0, 0.0
    tck, _ = splprep([pts_x, pts_y], k=3, s=0)
    u_new  = np.linspace(0, 1, num_frames)
    sx, sy = splev(u_new, tck)
    return np.column_stack((sx, sy))


# ============================================================
#  BLENDING
# ============================================================

def apply_blur(R_raw, sigma):
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(R_raw, (ksize | 1, ksize | 1), sigma)

def screen_blend_scalar(T_norm, R_norm, alpha, beta):
    T_w = T_norm * alpha
    R_w = R_norm * beta
    I   = T_w + R_w - T_w * R_w
    return np.clip(I, 0.0, 1.0), T_w, R_w


# ============================================================
#  DVS-VOLTMETER INTEGRATION
# ============================================================

def write_voltmeter_frames(frames_bgr, fps, scene_folder):
    os.makedirs(scene_folder, exist_ok=True)
    dt_us = int(1e6 / fps)
    lines = []
    for idx, frame in enumerate(frames_bgr):
        abs_path = os.path.join(scene_folder, f"frame_{idx:06d}.png")
        cv2.imwrite(abs_path, frame)
        lines.append(f"{abs_path} {idx * dt_us}\n")
    with open(os.path.join(scene_folder, "info.txt"), "w") as f:
        f.writelines(lines)

def patch_voltmeter_config(in_path, out_path, camera_type):
    config_path = os.path.join(DVS_VOLTMETER_PATH, "src", "config.py")
    with open(config_path, "r") as f: content = f.read()
    content = re.sub(r"^(__C\.DIR\.IN_PATH\s*=\s*).*$", f"__C.DIR.IN_PATH = '{in_path.rstrip('/') + '/'}'", content, flags=re.MULTILINE)
    content = re.sub(r"^(__C\.DIR\.OUT_PATH\s*=\s*).*$", f"__C.DIR.OUT_PATH = '{out_path.rstrip('/') + '/'}'", content, flags=re.MULTILINE)
    content = re.sub(r"^(__C\.SENSOR\.CAMERA_TYPE\s*=\s*).*$", f"__C.SENSOR.CAMERA_TYPE = '{camera_type}'", content, flags=re.MULTILINE)
    with open(config_path, "w") as f: f.write(content)

def run_dvs_voltmeter(scene_folder, voltmeter_out_dir):
    os.makedirs(voltmeter_out_dir, exist_ok=True)
    config_path   = os.path.join(DVS_VOLTMETER_PATH, "src", "config.py")
    backup_config = config_path + ".bak"
    shutil.copy(config_path, backup_config)
    try:
        patch_voltmeter_config(os.path.dirname(scene_folder.rstrip("/")), voltmeter_out_dir, DVS_VOLTMETER_CAMERA)
        subprocess.run(["conda", "run", "--no-capture-output", "-n", DVS_VOLTMETER_CONDA_ENV, "python3", "main.py"], 
                       check=True, cwd=DVS_VOLTMETER_PATH)
    finally:
        shutil.copy(backup_config, config_path)
        os.remove(backup_config)

def load_events_voltmeter(out_dir, scene_name, W, H):
    candidates = [f"{scene_name}.txt", f"{scene_name}.npy", "events.npy", "events.txt"]
    events_path = next((os.path.join(out_dir, p) for p in candidates if os.path.exists(os.path.join(out_dir, p))), None)

    if not events_path: return None
    events = np.load(events_path) if events_path.endswith(".npy") else np.loadtxt(events_path)
    if events.size == 0: return None
    if events.ndim == 1: events = np.expand_dims(events, axis=0)

    timestamps = events[:, 0].astype(np.float64)
    x          = events[:, 1].astype(np.int32)
    y          = events[:, 2].astype(np.int32)
    polarity   = 2 * events[:, 3].astype(np.int8) - 1

    if timestamps.max() > 1e4: timestamps *= 1e-6
    return timestamps, x, y, polarity, W, H


# ============================================================
#  VOXEL GRID PROCESSING
# ============================================================

def create_voxel_grid_soft(timestamps, x, y, polarity, t_start, t_end, W, H, num_bins):
    mask = (timestamps >= t_start) & (timestamps < t_end)
    ts_s, x_s, y_s, pol_s = timestamps[mask], x[mask], y[mask], polarity[mask]

    voxel = np.zeros((num_bins, H, W), dtype=np.float32)
    if ts_s.size == 0: return voxel

    t_min, t_max = ts_s.min(), ts_s.max()
    delta_t = max(t_max - t_min, 1e-6)

    ts  = (num_bins - 1) * (ts_s - t_min) / delta_t
    tis = ts.astype(np.int32)
    dts = ts - tis

    voxel_flat = voxel.ravel()
    for offset, weight in [(0, 1.0 - dts), (1, dts)]:
        valid = (tis + offset) < num_bins
        idx = x_s[valid] + y_s[valid] * W + (tis[valid] + offset) * W * H
        np.add.at(voxel_flat, idx, (pol_s * weight)[valid])

    voxel = voxel_flat.reshape((num_bins, H, W))
    p99 = np.percentile(np.abs(voxel), 99) or np.max(np.abs(voxel))
    if p99 > 0: voxel = np.clip(voxel, -p99, p99) / p99

    return voxel

def save_sum_png(voxel_grid, save_path):
    max_pos = np.max(np.maximum(voxel_grid, 0), axis=0)
    max_neg = np.max(np.maximum(-voxel_grid, 0), axis=0)
    img = np.zeros((voxel_grid.shape[1], voxel_grid.shape[2], 3), dtype=np.uint8)
    
    is_pos = (max_pos > 0.05) & (max_pos >= max_neg)
    is_neg = (max_neg > 0.05) & (max_neg > max_pos)
    img[is_pos] = [255, 0, 0]
    img[is_neg] = [0, 255, 0]
    plt.imsave(save_path, img)


# ============================================================
#  MAIN PIPELINE
# ============================================================

def process_sample(sample, coco_paths, rng):
    sample_id  = sample["sample_id"]
    out_dir    = os.path.join(OUT_ROOT, sample_id)

    if os.path.exists(os.path.join(out_dir, f"{sample_id}_E_voxel.npy")):
        return "skipped"

    T_img = cv2.imread(sample["path_T"])
    if T_img is None: return "error"

    W_tgt, H_tgt = TARGET_SIZE[0] + 2 * MARGIN, TARGET_SIZE[1] + 2 * MARGIN
    T_raw = resize_center_crop(T_img, W_tgt, H_tgt).astype(np.float32)

    # Stage 1: R Quality
    R_raw = None
    for _ in range(MAX_R_ATTEMPTS):
        R_img = cv2.imread(sample["path_R"])
        if R_img is not None:
            R_candidate = resize_center_crop(R_img, W_tgt, H_tgt).astype(np.float32)
            R_blurred = apply_blur(R_candidate, sample["blur_sigma"])
            R_blurred[cv2.cvtColor(R_blurred.astype(np.uint8), cv2.COLOR_BGR2GRAY) < 10] = 0
            
            if check_R_quality(R_blurred)[0]:
                R_raw = R_blurred
                break
        sample["path_R"] = coco_paths[rng.randint(0, len(coco_paths) - 1)]

    if R_raw is None: return "skipped"

    # Stage 2: Trajectory
    trajectory = None
    for _ in range(MAX_TRAJ_ATTEMPTS):
        candidate = generate_spline_trajectory(NUM_FRAMES)
        if check_trajectory_speed(candidate, TARGET_FRAME, VOXEL_N_FRAMES)[0] and \
           check_R_in_crop(candidate, TARGET_FRAME, MARGIN, R_raw, W_tgt, H_tgt)[0]:
            trajectory = candidate
            break

    if trajectory is None: return "skipped"

    os.makedirs(out_dir, exist_ok=True)
    height, width = T_raw.shape[:2]
    W_crop, H_crop = width - 2 * MARGIN, height - 2 * MARGIN
    T_norm = T_raw / 255.0

    with tempfile.TemporaryDirectory() as temp_dir:
        all_frames = []
        for i in range(NUM_FRAMES):
            dx, dy = trajectory[i]
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            R_warped = cv2.warpAffine(R_raw, M, (width, height))
            
            I_norm, T_w, R_w = screen_blend_scalar(T_norm, R_warped / 255.0, sample["alpha"], sample["beta"])
            I_crop = np.clip(I_norm * 255.0, 0, 255).astype(np.uint8)[MARGIN:height - MARGIN, MARGIN:width - MARGIN]
            all_frames.append(I_crop)

            if i == TARGET_FRAME:
                T_gt_crop    = np.clip(T_w * 255.0, 0, 255)[MARGIN:height-MARGIN, MARGIN:width-MARGIN].astype(np.uint8)
                R_wc_crop    = np.clip(R_w * 255.0, 0, 255)[MARGIN:height-MARGIN, MARGIN:width-MARGIN].astype(np.uint8)
                I_mixed_crop = I_crop.copy()

        if not diagnose_frame_diff(all_frames, TARGET_FRAME)[0]:
            shutil.rmtree(out_dir)
            return "skipped"

        # DVS-Voltmeter Simulation
        vm_scene_folder = os.path.join(temp_dir, "vm_input", f"{sample_id}_vm")
        vm_tmp = os.path.join(temp_dir, "vm_out")
        
        write_voltmeter_frames(all_frames, FPS, vm_scene_folder)
        try:
            run_dvs_voltmeter(vm_scene_folder, vm_tmp)
            data_vm = load_events_voltmeter(vm_tmp, f"{sample_id}_vm", W_crop, H_crop)
        except subprocess.CalledProcessError:
            shutil.rmtree(out_dir)
            return "error"

        if not data_vm:
            shutil.rmtree(out_dir)
            return "skipped"

        # Save assets
        cv2.imwrite(os.path.join(out_dir, f"{sample_id}_T_gt.jpg"), T_gt_crop)
        cv2.imwrite(os.path.join(out_dir, f"{sample_id}_I_mixed.jpg"), I_mixed_crop)
        cv2.imwrite(os.path.join(out_dir, f"{sample_id}_R_warped.jpg"), R_wc_crop)

        t_center = TARGET_FRAME / FPS
        ts, x, y, pol, _, _ = data_vm
        voxel = create_voxel_grid_soft(ts, x, y, pol, t_center - VOXEL_N_FRAMES / FPS, t_center + VOXEL_N_FRAMES / FPS, W_crop, H_crop, VOXEL_NUM_BINS)
        
        if not check_case_b(voxel, T_gt_crop, I_mixed_crop)[0]:
            shutil.rmtree(out_dir)
            return "skipped"

        np.save(os.path.join(out_dir, f"{sample_id}_E_voxel.npy"), voxel)
        save_sum_png(voxel, os.path.join(out_dir, f"{sample_id}_E_sum.png"))
        
        with open(os.path.join(out_dir, f"{sample_id}_meta.txt"), "w") as f:
            f.write(f"alpha={sample['alpha']:.6f}\nbeta={sample['beta']:.6f}\nblur_sigma={sample['blur_sigma']:.4f}\n")

    return "ok"

def main():
    print(f"Collecting COCO images from {COCO_ROOT}...")
    coco_paths = collect_image_paths(COCO_ROOT)
    if not coco_paths: return

    samples = build_sample_list(N_SAMPLES, coco_paths, seed=RANDOM_SEED)
    rng = random.Random(RANDOM_SEED + 99)

    skipped, failed, ok = 0, 0, 0
    for idx, sample in enumerate(samples, 1):
        print(f"[{idx}/{N_SAMPLES}] Processing {sample['sample_id']}...")
        try:
            res = process_sample(sample, coco_paths, rng)
            if res == "skipped": skipped += 1
            elif res == "error": failed += 1
            else: ok += 1
        except Exception as e:
            print(f"  Error: {e}")
            failed += 1

    print(f"\nDone! OK: {ok} | Skipped: {skipped} | Failed: {failed}")

if __name__ == "__main__":
    main()




"""
python generate_training_dataset_COCO.py

"""