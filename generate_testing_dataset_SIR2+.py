"""
Event-Guided SIRR: Synthetic Dataset Generation (SIR2+ Extension)
=================================================================
Extends the original SIR2+ dataset with synchronized event data.
Features:
- Alpha/beta estimation from existing blended mixtures via least-squares.
- Slingshot spline trajectories to guarantee high-speed motion at the center.
- Reflection padding approach to allow large displacements without border artifacts.
- Synchronised event voxel grid generation using DVS-Voltmeter.
"""

"""
FOR PEOPLE CLONING THE REPOSITORY: 
PLEASE MAKE SURE TO HAVE INSTALLED THE DVS VOLTMETER AND TO PASS THE CORRECT PATHS INSIDE THIS CODE.
SAME THING FOR THE SIR2+ DATASET.
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
from collections import defaultdict
from scipy.interpolate import splprep, splev

# ============================================================
#  CONFIGURATION
# ============================================================

DATASET_ROOTS = [
    "/medias/db/ImagingSecurity_misc/SIRR_Datasets/SIR2/SIR2+/Outdoor",
    "/medias/db/ImagingSecurity_misc/SIRR_Datasets/SIR2/SIR2+/Indoor"
]
OUT_ROOT = "./data/synthetic_SIR2/dvs"

N_TRAJECTORIES = 1
ALPHA_FALLBACK = 0.80
BETA_FALLBACK  = 0.50

# --- Rendering ---
NUM_FRAMES   = 300
FPS          = 30
TARGET_FRAME = 150   # Midpoint for synchronous voxel extraction

MAX_DISP     = 60    # Max trajectory displacement (px)
MARGIN       = 8     # Anti-aliasing crop margin (px)
R_PAD_SAFETY = 8     # Safety padding added to MAX_DISP (px)

RANDOM_SEED = 42

# --- Quality Filters ---
MIN_TRAJ_SPEED_PX        = 2.0
MIN_R_CROP_FRAC          = 0.05
MIN_MEAN_DIFF            = 0.25
MIN_DIFF_FOR_VOXEL_CHECK = 40.0
MIN_VOXEL_DENSITY        = 0.010
MAX_TRAJ_ATTEMPTS        = 10

# --- DVS-Voltmeter ---
DVS_VOLTMETER_PATH      = "/medias/db/ImagingSecurity_misc/melcarne/Event_simulators/DVS-Voltmeter"
DVS_VOLTMETER_CONDA_ENV = "dvs-voltmeter"
DVS_VOLTMETER_CAMERA    = "DVSCustomSIRR"

# --- Voxel ---
VOXEL_N_FRAMES = 0.5
VOXEL_NUM_BINS = 5

# --- Skip Reasons ---
SKIP_TRAJ_SPEED   = "trajectory_too_slow"
SKIP_R_CROP       = "R_out_of_crop"
SKIP_TRAJ_FAILED  = "trajectory_exhausted"
SKIP_FRAME_DIFF   = "frame_diff_too_low"
SKIP_CASE_B       = "case_B_empty_voxel"
SKIP_ALREADY_DONE = "already_processed"


# ============================================================
#  DATASET DISCOVERY
# ============================================================

def discover_scenes(dataset_roots):
    scenes, seen = [], set()
    for dataset_root in dataset_roots:
        split_tag = "Outdoor" if "Outdoor" in dataset_root else "Indoor"
        for dirpath, dirnames, filenames in os.walk(dataset_root):
            dirnames.sort()
            bg_name = "background.png" if "background.png" in filenames else ("groundtruth.png" if "groundtruth.png" in filenames else None)
            if not bg_name or "reflection.png" not in filenames:
                continue

            path_mix = next((os.path.join(dirpath, m) for m in ["mixture.png", "blended.png"] if m in filenames), None)
            scene_id = f"{split_tag}_" + os.path.relpath(dirpath, dataset_root).replace(os.sep, "_")
            
            if scene_id not in seen:
                seen.add(scene_id)
                scenes.append((scene_id, os.path.join(dirpath, bg_name), os.path.join(dirpath, "reflection.png"), path_mix, split_tag))

    def sort_key(item):
        return [(int("".join(filter(str.isdigit, p))) if any(c.isdigit() for c in p) else float("inf"), p) for p in item[0].split("_")]
    
    return sorted(scenes, key=sort_key)


# ============================================================
#  ALPHA/BETA ESTIMATION & BLENDING
# ============================================================

def estimate_alpha_beta_scene(path_T, path_R, path_mixture):
    """Least-squares fit of the linear model I ~ alpha*T + beta*R."""
    T, R, I = cv2.imread(path_T), cv2.imread(path_R), cv2.imread(path_mixture)
    if any(img is None for img in [T, R, I]):
        return ALPHA_FALLBACK, BETA_FALLBACK

    T, R, I = T.astype(np.float32) / 255.0, R.astype(np.float32) / 255.0, I.astype(np.float32) / 255.0
    if R.shape[:2] != T.shape[:2]: R = cv2.resize(R, (T.shape[1], T.shape[0]))
    if I.shape[:2] != T.shape[:2]: I = cv2.resize(I, (T.shape[1], T.shape[0]))

    mask = (cv2.cvtColor((R * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) > 10) & \
           (cv2.cvtColor((T * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) < 245)

    if mask.sum() < 100: return ALPHA_FALLBACK, BETA_FALLBACK

    alphas, betas = [], []
    for c in range(3):
        res = np.linalg.lstsq(np.hstack([T[mask, c:c+1], R[mask, c:c+1]]), I[mask, c], rcond=None)[0]
        alphas.append(res[0]); betas.append(res[1])

    return float(np.clip(np.median(alphas), 0.50, 0.85)), float(np.clip(np.median(betas), 0.30, 0.65))

def screen_blend_scalar(T_norm, R_norm, alpha, beta):
    T_w, R_w = T_norm * alpha, R_norm * beta
    return np.clip(T_w + R_w - T_w * R_w, 0.0, 1.0), T_w, R_w

def pad_reflection(R_raw, pad):
    R_padded = cv2.copyMakeBorder(R_raw.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return R_padded.astype(np.float32) / 255.0, pad, pad


# ============================================================
#  TRAJECTORY & QUALITY FILTERS
# ============================================================

def generate_slingshot_trajectory(num_frames, max_displacement):
    """Time-Compressed Slingshot: forces high speed at the center."""
    safe_r = max_displacement * 0.8
    u_keys = [0.0, 0.20, 0.45, 0.50, 0.55, 0.80, 1.0]
    direction = np.random.choice([-1.0, 1.0])
    
    if np.random.choice(['x', 'y']) == 'x':
        pts_x = [0.0, np.random.uniform(-safe_r*0.4, safe_r*0.4), -direction*safe_r, 0.0, direction*safe_r, np.random.uniform(-safe_r*0.4, safe_r*0.4), 0.0]
        pts_y = [0.0, np.random.uniform(-safe_r*0.4, safe_r*0.4), np.random.uniform(-safe_r*0.4, safe_r*0.4), np.random.uniform(-safe_r*0.1, safe_r*0.1), np.random.uniform(-safe_r*0.4, safe_r*0.4), np.random.uniform(-safe_r*0.4, safe_r*0.4), 0.0]
    else:
        pts_x = [0.0, np.random.uniform(-safe_r*0.4, safe_r*0.4), np.random.uniform(-safe_r*0.4, safe_r*0.4), np.random.uniform(-safe_r*0.1, safe_r*0.1), np.random.uniform(-safe_r*0.4, safe_r*0.4), np.random.uniform(-safe_r*0.4, safe_r*0.4), 0.0]
        pts_y = [0.0, np.random.uniform(-safe_r*0.4, safe_r*0.4), -direction*safe_r, 0.0, direction*safe_r, np.random.uniform(-safe_r*0.4, safe_r*0.4), 0.0]

    tck, _ = splprep([pts_x, pts_y], u=u_keys, k=3, s=0)
    sx, sy = splev(np.linspace(0, 1, num_frames), tck)
    return np.column_stack((sx, sy))

def guaranteed_speed_trajectory(num_frames, max_displacement, target_frame, n_frames_window):
    for _ in range(MAX_TRAJ_ATTEMPTS):
        traj = generate_slingshot_trajectory(num_frames, max_displacement)
        f_start, f_end = max(0, target_frame - int(math.ceil(n_frames_window))), min(num_frames - 1, target_frame + int(math.ceil(n_frames_window)))
        speeds = np.sqrt(np.sum(np.diff(traj[f_start:f_end + 1], axis=0)**2, axis=1))
        speed = speeds.mean() if len(speeds) > 0 else 0.0
        if speed >= MIN_TRAJ_SPEED_PX:
            return traj, float(speed)

    # Deterministic fallback
    safe_r = max_displacement * 0.8
    tck, _ = splprep([[0.0, 0.0, -safe_r, 0.0, safe_r, 0.0, 0.0], [0.0]*7], u=[0.0, 0.20, 0.45, 0.50, 0.55, 0.80, 1.0], k=3, s=0)
    sx, sy = splev(np.linspace(0, 1, num_frames), tck)
    traj = np.column_stack((sx, sy))
    speeds = np.sqrt(np.sum(np.diff(traj[max(0, target_frame-2):target_frame+3], axis=0)**2, axis=1))
    return traj, float(speeds.mean()) if len(speeds) > 0 else 0.0

def check_R_in_crop(trajectory, target_frame, margin, R_padded_norm, r_pad, width, height):
    dx, dy = trajectory[target_frame]
    M = np.float32([[1, 0, dx - r_pad], [0, 1, dy - r_pad]])
    R_warped = cv2.warpAffine((R_padded_norm * 255).astype(np.uint8), M, (width, height))
    R_crop = R_warped[margin:height - margin, margin:width - margin]
    frac_visible = float((cv2.cvtColor(R_crop, cv2.COLOR_BGR2GRAY) > 10).mean())
    return frac_visible >= MIN_R_CROP_FRAC, frac_visible

def diagnose_frame_diff(frames, target_frame):
    diff = (np.abs(frames[target_frame].astype(np.float32) - frames[target_frame-1].astype(np.float32)) + 
            np.abs(frames[target_frame+1].astype(np.float32) - frames[target_frame].astype(np.float32))) / 2.0
    mean_diff = float(diff.mean(axis=2).mean())
    return mean_diff >= MIN_MEAN_DIFF, mean_diff

def check_case_b(voxel, T_gt, I_mixed):
    diff_mean = float(np.mean(np.abs(cv2.cvtColor(I_mixed, cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(T_gt, cv2.COLOR_BGR2GRAY).astype(np.float32))))
    if diff_mean <= MIN_DIFF_FOR_VOXEL_CHECK: return True, "ok (low diff)"
    voxel_density = float(np.mean(np.max(np.abs(voxel), axis=0) > 0.05))
    return voxel_density >= MIN_VOXEL_DENSITY, f"density={voxel_density:.4f}"


# ============================================================
#  DVS-VOLTMETER & VOXEL INTEGRATION
# ============================================================

def run_dvs_voltmeter(frames_bgr, fps, vm_tmp_dir):
    os.makedirs(vm_tmp_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as vm_input:
        dt_us = int(1e6 / fps)
        with open(os.path.join(vm_input, "info.txt"), "w") as f:
            for idx, frame in enumerate(frames_bgr):
                path = os.path.join(vm_input, f"frame_{idx:06d}.png")
                cv2.imwrite(path, frame)
                f.write(f"{path} {idx * dt_us}\n")

        config_path = os.path.join(DVS_VOLTMETER_PATH, "src", "config.py")
        backup_config = config_path + ".bak"
        shutil.copy(config_path, backup_config)
        try:
            with open(config_path, "r") as f: content = f.read()
            content = re.sub(r"^(__C\.DIR\.IN_PATH\s*=\s*).*$", f"__C.DIR.IN_PATH = '{vm_input}/'", content, flags=re.MULTILINE)
            content = re.sub(r"^(__C\.DIR\.OUT_PATH\s*=\s*).*$", f"__C.DIR.OUT_PATH = '{vm_tmp_dir}/'", content, flags=re.MULTILINE)
            content = re.sub(r"^(__C\.SENSOR\.CAMERA_TYPE\s*=\s*).*$", f"__C.SENSOR.CAMERA_TYPE = '{DVS_VOLTMETER_CAMERA}'", content, flags=re.MULTILINE)
            with open(config_path, "w") as f: f.write(content)
            
            subprocess.run(["conda", "run", "--no-capture-output", "-n", DVS_VOLTMETER_CONDA_ENV, "python3", "main.py"], check=True, cwd=DVS_VOLTMETER_PATH)
        finally:
            shutil.copy(backup_config, config_path)
            os.remove(backup_config)

def load_events_voltmeter(out_dir, W, H):
    events_path = next((os.path.join(out_dir, p) for p in os.listdir(out_dir) if p.endswith(".npy") or p.endswith(".txt")), None)
    if not events_path: return None
    events = np.load(events_path) if events_path.endswith(".npy") else np.loadtxt(events_path)
    if events.size == 0: return None
    if events.ndim == 1: events = np.expand_dims(events, axis=0)

    ts = events[:, 0].astype(np.float64)
    if ts.max() > 1e4: ts *= 1e-6
    return ts, events[:, 1].astype(np.int32), events[:, 2].astype(np.int32), 2 * events[:, 3].astype(np.int8) - 1, W, H

def create_voxel_grid(data, t_center, fps, W, H):
    if not data: return None
    ts, x, y, pol, _, _ = data
    t_start, t_end = t_center - VOXEL_N_FRAMES / fps, t_center + VOXEL_N_FRAMES / fps
    mask = (ts >= t_start) & (ts < t_end)
    ts_s, x_s, y_s, pol_s = ts[mask], x[mask], y[mask], pol[mask]
    
    voxel = np.zeros((VOXEL_NUM_BINS, H, W), dtype=np.float32)
    if ts_s.size == 0: return voxel

    t_min, delta_t = ts_s.min(), max(ts_s.max() - ts_s.min(), 1e-6)
    ts_norm = (VOXEL_NUM_BINS - 1) * (ts_s - t_min) / delta_t
    tis, dts = ts_norm.astype(np.int32), ts_norm - ts_norm.astype(np.int32)

    voxel_flat = voxel.ravel()
    for offset, weight in [(0, 1.0 - dts), (1, dts)]:
        valid = (tis + offset) < VOXEL_NUM_BINS
        np.add.at(voxel_flat, x_s[valid] + y_s[valid] * W + (tis[valid] + offset) * W * H, (pol_s * weight)[valid])
    
    voxel = voxel_flat.reshape((VOXEL_NUM_BINS, H, W))
    p99 = np.percentile(np.abs(voxel), 99) or np.max(np.abs(voxel))
    return np.clip(voxel, -p99, p99) / p99 if p99 > 0 else voxel


# ============================================================
#  PIPELINE
# ============================================================

def process_scene(scene_id, path_T, path_R, path_mix, split_tag, skip_stats):
    T_img, R_img = cv2.imread(path_T), cv2.imread(path_R)
    if T_img is None or R_img is None: return 1

    T_raw = T_img.astype(np.float32)
    if R_img.shape[:2] != T_img.shape[:2]: R_img = cv2.resize(R_img, (T_img.shape[1], T_img.shape[0]))
    R_raw = R_img.astype(np.float32)
    R_raw[cv2.cvtColor(R_raw.astype(np.uint8), cv2.COLOR_BGR2GRAY) < 10] = 0

    alpha, beta = estimate_alpha_beta_scene(path_T, path_R, path_mix) if path_mix else (ALPHA_FALLBACK, BETA_FALLBACK)
    skip_stats["__beta_all__"].append(beta)

    skipped = 0
    for traj_idx in range(N_TRAJECTORIES):
        sample_id = f"{scene_id}_t{traj_idx:02d}"
        out_dvs = os.path.join(OUT_ROOT, split_tag, scene_id, sample_id)

        if os.path.exists(os.path.join(out_dvs, f"{sample_id}_E_voxel.npy")):
            skip_stats[SKIP_ALREADY_DONE].append(sample_id)
            continue

        r_pad = MAX_DISP + R_PAD_SAFETY
        R_padded_norm, cx_offset, cy_offset = pad_reflection(R_raw, r_pad)
        H_crop, W_crop = T_raw.shape[0] - 2 * MARGIN, T_raw.shape[1] - 2 * MARGIN

        # Stage 1: Trajectory
        trajectory, _ = guaranteed_speed_trajectory(NUM_FRAMES, MAX_DISP, TARGET_FRAME, VOXEL_N_FRAMES)
        if not check_R_in_crop(trajectory, TARGET_FRAME, MARGIN, R_padded_norm, r_pad, T_raw.shape[1], T_raw.shape[0])[0]:
            skip_stats[SKIP_R_CROP].append(sample_id)
            skipped += 1
            continue

        os.makedirs(out_dvs, exist_ok=True)
        all_frames = []
        for i in range(NUM_FRAMES):
            dx, dy = trajectory[i]
            M = np.float32([[1, 0, dx - cx_offset], [0, 1, dy - cy_offset]])
            R_warped = cv2.warpAffine(R_padded_norm, M, (T_raw.shape[1], T_raw.shape[0]))
            
            I_norm, T_w, R_w = screen_blend_scalar(T_raw / 255.0, R_warped, alpha, beta)
            I_crop = np.clip(I_norm * 255.0, 0, 255).astype(np.uint8)[MARGIN:-MARGIN, MARGIN:-MARGIN]
            all_frames.append(I_crop)

            if i == TARGET_FRAME:
                T_gt_crop = np.clip(T_w * 255.0, 0, 255)[MARGIN:-MARGIN, MARGIN:-MARGIN].astype(np.uint8)
                R_wc_crop = np.clip(R_w * 255.0, 0, 255)[MARGIN:-MARGIN, MARGIN:-MARGIN].astype(np.uint8)
                I_mixed_crop = I_crop.copy()

        # Stage 2: Diagnostic
        ok_diff, diff_val = diagnose_frame_diff(all_frames, TARGET_FRAME)
        if not ok_diff:
            shutil.rmtree(out_dvs)
            skip_stats[SKIP_FRAME_DIFF].append(sample_id)
            skipped += 1
            continue

        # Simulation
        with tempfile.TemporaryDirectory() as vm_tmp:
            try:
                run_dvs_voltmeter(all_frames, FPS, vm_tmp)
                data_vm = load_events_voltmeter(vm_tmp, W_crop, H_crop)
            except subprocess.CalledProcessError:
                shutil.rmtree(out_dvs)
                skipped += 1
                continue

        voxel = create_voxel_grid(data_vm, TARGET_FRAME / FPS, FPS, W_crop, H_crop)
        if voxel is None or not check_case_b(voxel, T_gt_crop, I_mixed_crop)[0]:
            shutil.rmtree(out_dvs)
            skip_stats[SKIP_CASE_B].append(sample_id)
            skipped += 1
            continue

        # Save
        cv2.imwrite(os.path.join(out_dvs, f"{sample_id}_T_gt.jpg"), T_gt_crop)
        cv2.imwrite(os.path.join(out_dvs, f"{sample_id}_I_mixed.jpg"), I_mixed_crop)
        cv2.imwrite(os.path.join(out_dvs, f"{sample_id}_R_warped.jpg"), R_wc_crop)
        np.save(os.path.join(out_dvs, f"{sample_id}_E_voxel.npy"), voxel)
        
        with open(os.path.join(out_dvs, f"{sample_id}_meta.txt"), "w") as f:
            f.write(f"alpha={alpha:.3f}\nbeta={beta:.3f}\nframe_diff_mean={diff_val:.3f}\n")

    return skipped

def main():
    np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)
    scenes = discover_scenes(DATASET_ROOTS)
    print(f"Discovered {len(scenes)} scenes. Starting Generation...")

    skip_stats, failed, skipped = defaultdict(list), [], 0
    for idx, (s_id, p_T, p_R, p_mix, tag) in enumerate(scenes, 1):
        print(f"[{idx}/{len(scenes)}] Processing {s_id}...")
        try:
            skipped += process_scene(s_id, p_T, p_R, p_mix, tag, skip_stats)
        except Exception as e:
            print(f"  Error: {e}"); failed.append(s_id)

    print(f"\nDone! Processed: {len(scenes)} | Skipped: {skipped} | Errors: {len(failed)}")

if __name__ == "__main__":
    main()