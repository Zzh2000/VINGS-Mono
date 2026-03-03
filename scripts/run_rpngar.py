import os
import sys
import yaml
import numpy as np
from glob import glob
import socket
import shutil
from math import exp
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ---------------------------------------------------------------------------
# Image quality metrics (PSNR / SSIM / LPIPS)
# ---------------------------------------------------------------------------

def psnr(img1, img2):
    """img1, img2: (B, 3, H, W) float tensors in [0, 1]"""
    import torch
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def _gaussian_window(window_size, sigma):
    import torch
    gauss = torch.tensor([
        exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def ssim(img1, img2, window_size=11):
    """img1, img2: (B, 3, H, W) float tensors in [0, 1]. Returns scalar."""
    import torch
    import torch.nn.functional as F
    channel = img1.size(1)
    _1d = _gaussian_window(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous().to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    s2 = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    s12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / (
               (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean()


# Per-sequence fixed eval indices (random fallback when key absent)
RPNGAR_EVAL_INDICES = {
    'table_01': [1719, 1832, 703, 1775, 2234, 963, 2242, 1108, 195, 224, 1604, 1567, 2473, 1611, 2085, 1423, 33, 385, 1187, 53, 414, 1236, 1192, 1559, 997, 1614, 1418, 1225, 2184, 1804, 1256, 2297, 135, 651, 379, 1633, 1017, 333, 1875, 674, 2405, 418, 1485, 670, 271, 1818, 923, 2368, 1927, 2106],
    'table_02': [1013, 116, 2374, 2193, 671, 2874, 585, 738, 1587, 308, 1372, 2278, 1919, 891, 146, 918, 631, 2836, 1433, 142, 1258, 1306, 1629, 2696, 623, 1839, 2017, 214, 80, 962, 1008, 2168, 496, 2172, 111, 2711, 1924, 2615, 1536, 2578, 1139, 1614, 2295, 1953, 1136, 39, 829, 1702, 2871, 2135],
    'table_03': [6372, 2923, 5586, 1769, 1167, 5150, 1199, 5054, 2105, 2252, 5866, 3487, 6735, 1404, 2521, 2359, 4807, 4297, 851, 449, 3810, 308, 1328, 3831, 3295, 1091, 5004, 3840, 797, 521, 2509, 6883, 5413, 2085, 183, 2070, 1156, 6332, 3634, 1593, 4629, 1414, 6747, 6137, 6573, 1007, 4200, 2092, 1076, 6472],
    'table_04': [1116, 4263, 4884, 4484, 4668, 2791, 4439, 1938, 1467, 4995, 5779, 2651, 1221, 5376, 2041, 4292, 5939, 5314, 1553, 4250, 3590, 3295, 5818, 5671, 397, 4849, 768, 5961, 4865, 4555, 3685, 5900, 841, 4698, 3676, 624, 4370, 4376, 3743, 5552, 4297, 5959, 97, 3039, 1156, 506, 953, 1492, 4727, 861],
    'table_05': [308, 3187, 3580, 1988, 57, 5436, 4656, 5673, 5362, 2593, 1172, 3672, 4565, 3179, 3980, 4546, 1028, 3429, 3670, 2126, 870, 1473, 3863, 2339, 4590, 2907, 1114, 4596, 4073, 3761, 4945, 1793, 1896, 4452, 3524, 2135, 1361, 1975, 1909, 352, 79, 88, 3096, 759, 3166, 601, 5162, 2633, 3435, 1776],
    'table_06': [2571, 2379, 2424, 1602, 1466, 427, 2480, 151, 1361, 2257, 1473, 836, 1467, 2508, 876, 2185, 1283, 1619, 1314, 461, 2229, 2225, 1771, 2212, 2601, 1028, 1011, 1001, 2734, 1447, 572, 2596, 2351, 2089, 116, 1854, 2707, 1811, 1009, 198, 1833, 1464, 1737, 373, 337, 1985, 2540, 1254, 628, 1503],
    'table_07': [2300, 177, 2914, 3228, 1357, 1087, 2339, 2764, 4643, 923, 2580, 4278, 2781, 4188, 4276, 1673, 1159, 1819, 426, 2631, 1315, 3223, 4263, 3897, 1011, 1009, 3507, 2896, 656, 1560, 3986, 1158, 3063, 2046, 1456, 2292, 1382, 4024, 4451, 179, 3475, 2997, 2201, 322, 842, 444, 2374, 2314, 4660, 4339],
    'table_08': [803, 4409, 2319, 2663, 590, 1967, 3544, 154, 1237, 6674, 593, 7541, 6301, 309, 5033, 295, 2960, 6929, 6214, 5350, 3551, 2857, 2693, 940, 8447, 7790, 6777, 6890, 5523, 1984, 8193, 2037, 2172, 1368, 6277, 6312, 6634, 2409, 7649, 3571, 2677, 3450, 6407, 5453, 5455, 3682, 7674, 5613, 6181, 5961],
}


def load_tum_trajectory(filepath):
    """Load TUM format trajectory: timestamp tx ty tz qx qy qz qw"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                data.append([float(x) for x in parts[:8]])
    return np.array(data)


def align_trajectories(est_traj, gt_traj, max_diff=0.01):
    """Align estimated trajectory to ground truth by timestamp matching."""
    aligned_est = []
    aligned_gt = []

    gt_times = gt_traj[:, 0]
    for est in est_traj:
        t = est[0]
        idx = np.argmin(np.abs(gt_times - t))
        if np.abs(gt_times[idx] - t) < max_diff:
            aligned_est.append(est)
            aligned_gt.append(gt_traj[idx])

    return np.array(aligned_est), np.array(aligned_gt)


def umeyama_alignment(est_xyz, gt_xyz, with_scale=True):
    """Align est to gt using Umeyama's method. Returns R, t, s."""
    mu_est = est_xyz.mean(axis=0)
    mu_gt = gt_xyz.mean(axis=0)

    est_centered = est_xyz - mu_est
    gt_centered = gt_xyz - mu_gt

    H = est_centered.T @ gt_centered / len(est_xyz)
    U, S, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T

    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T

    if with_scale:
        var_est = np.sum(est_centered ** 2) / len(est_xyz)
        scale = np.sum(S) / var_est
    else:
        scale = 1.0

    t_align = mu_gt - scale * R_align @ mu_est

    return R_align, t_align, scale


def plot_trajectory_2d(est_traj, gt_traj, title, output_path):
    """Plot 2D trajectory comparison (top-down view)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Align trajectories
    aligned_est, aligned_gt = align_trajectories(est_traj, gt_traj)
    if len(aligned_est) < 3:
        print(f"Not enough aligned points for {title}")
        return

    est_xyz = aligned_est[:, 1:4]
    gt_xyz = aligned_gt[:, 1:4]

    # Umeyama alignment
    R_align, t_align, scale = umeyama_alignment(est_xyz, gt_xyz, with_scale=True)
    est_aligned = (scale * (R_align @ est_xyz.T).T + t_align)

    # Compute APE
    ape = np.linalg.norm(est_aligned - gt_xyz, axis=1)

    # XY plot (top-down)
    ax = axes[0]
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(est_aligned[:, 0], est_aligned[:, 1], 'r--', label='Estimated', linewidth=2)
    ax.scatter(gt_xyz[0, 0], gt_xyz[0, 1], c='g', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], c='m', s=100, marker='s', zorder=5, label='End')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Top-Down View (XY)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # XZ plot (side view)
    ax = axes[1]
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(est_aligned[:, 0], est_aligned[:, 2], 'r--', label='Estimated', linewidth=2)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('Side View (XZ)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # APE over time
    ax = axes[2]
    timestamps = aligned_est[:, 0] - aligned_est[0, 0]  # Relative time
    ax.plot(timestamps, ape * 100, 'r-', linewidth=1.5)
    ax.axhline(y=np.mean(ape) * 100, color='b', linestyle='--', label=f'Mean: {np.mean(ape)*100:.2f} cm')
    ax.axhline(y=np.sqrt(np.mean(ape**2)) * 100, color='g', linestyle='--', label=f'RMSE: {np.sqrt(np.mean(ape**2))*100:.2f} cm')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('APE [cm]')
    ax.set_title('Absolute Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{title}\nScale: {scale:.4f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plot: {output_path}")


def plot_trajectory_3d(est_traj, gt_traj, title, output_path):
    """Plot 3D trajectory comparison."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Align trajectories
    aligned_est, aligned_gt = align_trajectories(est_traj, gt_traj)
    if len(aligned_est) < 3:
        print(f"Not enough aligned points for {title}")
        return

    est_xyz = aligned_est[:, 1:4]
    gt_xyz = aligned_gt[:, 1:4]

    # Umeyama alignment
    R_align, t_align, scale = umeyama_alignment(est_xyz, gt_xyz, with_scale=True)
    est_aligned = (scale * (R_align @ est_xyz.T).T + t_align)

    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(est_aligned[:, 0], est_aligned[:, 1], est_aligned[:, 2], 'r--', label='Estimated', linewidth=2)
    ax.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], c='g', s=100, marker='o', label='Start')
    ax.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], gt_xyz[-1, 2], c='m', s=100, marker='s', label='End')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'{title}\nScale: {scale:.4f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D trajectory plot: {output_path}")


def plot_summary(results, output_path):
    """Plot summary bar chart of all sequences."""
    valid_results = [r for r in results if r['ate'] is not None]
    if not valid_results:
        print("No valid results to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    seq_names = [r['seq'] for r in valid_results]
    ate_values = [r['ate'] * 100 for r in valid_results]
    scale_errors = [abs(1 - r['scale']) * 100 if r['scale'] else 0 for r in valid_results]

    x = np.arange(len(seq_names))
    width = 0.6

    # ATE bar chart
    ax = axes[0]
    bars = ax.bar(x, ate_values, width, color='steelblue', edgecolor='black')
    ax.axhline(y=np.mean(ate_values), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ate_values):.2f} cm')
    ax.set_xlabel('Sequence')
    ax.set_ylabel('ATE RMSE [cm]')
    ax.set_title('Absolute Trajectory Error')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, ate_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Scale error bar chart
    ax = axes[1]
    bars = ax.bar(x, scale_errors, width, color='coral', edgecolor='black')
    ax.axhline(y=np.mean(scale_errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scale_errors):.2f}%')
    ax.set_xlabel('Sequence')
    ax.set_ylabel('Scale Error [%]')
    ax.set_title('Scale Error')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, scale_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('RPNG-AR Evaluation Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot: {output_path}")


def matrix_to_tum(timestamp, T):
    """Convert 4x4 transformation matrix to TUM format (t tx ty tz qx qy qz qw)."""
    tx, ty, tz = T[:3, 3]
    rot = R.from_matrix(T[:3, :3])
    qx, qy, qz, qw = rot.as_quat()  # scipy returns [x, y, z, w]
    return [timestamp, tx, ty, tz, qx, qy, qz, qw]


def combine_poses_to_tum(droid_c2w_dir, output_file, rgb_dir=None):
    """Combine individual pose files into TUM trajectory format.

    Args:
        droid_c2w_dir: Directory containing per-keyframe pose files (XXXXXX.Y.txt)
        output_file: Output TUM trajectory file
        rgb_dir: Optional RGB directory to get actual timestamps from filenames
    """
    pose_files = sorted(glob(os.path.join(droid_c2w_dir, '*.txt')))
    if not pose_files:
        return None

    # # If rgb_dir provided, get actual timestamps from image filenames
    # timestamp_map = {}
    # if rgb_dir and os.path.isdir(rgb_dir):
    #     rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')])
    #     for idx, rf in enumerate(rgb_files):
    #         # Timestamp from filename (nanoseconds to seconds)
    #         try:
    #             breakpoint()
    #             ts = float(os.path.splitext(rf)[0]) / 1e9
    #             timestamp_map[idx] = ts
    #         except:
    #             timestamp_map[idx] = float(idx)

    tum_data = []
    for pf in pose_files:
        # Extract frame index from filename (format: XXXXXX.Y.txt)
        basename = os.path.basename(pf)
        parts = basename.replace('.txt', '').split('.')
        frame_idx = int(parts[0])

        # # Get timestamp from map or use frame index
        # if frame_idx in timestamp_map:
        #     timestamp = timestamp_map[frame_idx]
        # else:
        #     timestamp = float(frame_idx)
        # breakpoint()
        timestamp = float(os.path.basename(pf)[:-4])
        try:
            T = np.loadtxt(pf)
            if T.shape == (4, 4):
                tum_data.append(matrix_to_tum(timestamp, T))
        except Exception as e:
            print(f"Error loading {pf}: {e}")
        # breakpoint()
    if not tum_data:
        return None

    tum_data = sorted(tum_data, key=lambda x: x[0])
    np.savetxt(output_file, tum_data, fmt='%.9f')
    return output_file


def render_and_eval_sequence(seq_path, output_folder, use_full_traj_for_poses=True, n_eval=50):
    """Load saved 2DGS model, render at n_eval random frames, compute PSNR/SSIM/LPIPS.

    Args:
        seq_path            : path to the dataset sequence root
        output_folder       : top-level output folder (same as passed to run_sequence)
        use_full_traj_for_poses : prefer traj_full.txt over traj_combined.txt
        n_eval              : number of frames to evaluate (random subset)

    Returns:
        dict with keys 'psnr', 'ssim', 'lpips', 'n'  (or None on failure)
    """
    import importlib, torch, torch.nn.functional as F

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from gaussian.gaussian_model import GaussianModel
    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

    seq_name   = os.path.basename(seq_path)
    seq_output = os.path.join(output_folder, seq_name)
    subdirs    = sorted([d for d in glob(os.path.join(seq_output, '*')) if os.path.isdir(d)])

    # ---- locate run directory (contains a non-empty ply/ with *_2dgs.ply) --
    run_dir = None
    for d in [seq_output] + subdirs:
        if glob(os.path.join(d, 'ply', '*_2dgs.ply')):
            run_dir = d
            break
    if run_dir is None:
        print(f"[render_eval] No ply/ directory for {seq_name}, skipping.")
        return None

    # ---- find .ply checkpoint ---------------------------------------------
    ply_files = sorted(glob(os.path.join(run_dir, 'ply', '*_2dgs.ply')))
    if not ply_files:
        print(f"[render_eval] No *_2dgs.ply found under {run_dir}/ply/")
        return None
    ply_path = ply_files[-1]

    # ---- load rendering intrinsics ----------------------------------------
    intrinsic_path = os.path.join(run_dir, 'ply', 'intrinsic.yaml')
    if not os.path.exists(intrinsic_path):
        print(f"[render_eval] No intrinsic.yaml for {seq_name}")
        return None
    with open(intrinsic_path) as f:
        intr = yaml.safe_load(f)   # keys: fu, fv, cu, cv, H, W

    # ---- locate trajectory ------------------------------------------------
    traj_file = None
    pref = ['traj_full.txt', 'traj_combined.txt'] if use_full_traj_for_poses \
           else ['traj_combined.txt', 'traj_full.txt']
    for d in [seq_output] + subdirs:
        for name in pref:
            c = os.path.join(d, name)
            if os.path.exists(c):
                traj_file = c
                break
        if traj_file:
            break
    if traj_file is None:
        print(f"[render_eval] No trajectory for {seq_name}")
        return None
    print(f"[render_eval] Using trajectory: {traj_file}")

    traj       = load_tum_trajectory(traj_file)   # (N, 8): ts tx ty tz qx qy qz qw
    traj_times = traj[:, 0]

    # ---- load config saved by the SLAM run --------------------------------
    cfg_path = os.path.join(run_dir, 'config.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg['dataset']['root']  = seq_path
    cfg['device']           = {'tracker': 'cuda:0', 'mapper': 'cuda:0'}

    # Render at full original resolution
    cfg_intr = cfg['intrinsic']
    intr = {
        'H':  cfg_intr['H'],
        'W':  cfg_intr['W'],
        'fu': cfg_intr['fu'],
        'fv': cfg_intr['fv'],
        'cu': cfg_intr['cu'],
        'cv': cfg_intr['cv'],
    }
    print(f"[render_eval] Rendering at full resolution: {intr['W']}x{intr['H']}")

    # ---- load dataset for timestamps / filepaths --------------------------
    get_dataset_fn = importlib.import_module(cfg['dataset']['module']).get_dataset
    dataset  = get_dataset_fn(cfg)
    n_frames = len(dataset)

    if seq_name in RPNGAR_EVAL_INDICES:
        eval_indices = [i for i in RPNGAR_EVAL_INDICES[seq_name] if i < n_frames]
        print(f"[render_eval] Using fixed eval indices for {seq_name} (n={len(eval_indices)})")
    else:
        rng          = np.random.default_rng(42)
        eval_indices = sorted(rng.choice(n_frames, size=min(n_eval, n_frames), replace=False).tolist())

    # ---- load Gaussian model ----------------------------------------------
    mapper = GaussianModel(cfg)
    mapper.load_ply_ckpt(ply_path)
    mapper.tfer.H  = intr['H']
    mapper.tfer.W  = intr['W']
    mapper.tfer.fu = intr['fu']
    mapper.tfer.fv = intr['fv']
    mapper.tfer.cu = intr['cu']
    mapper.tfer.cv = intr['cv']

    device   = 'cuda:0'
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    render_imgs_dir = os.path.join(seq_output, 'render_imgs')
    os.makedirs(render_imgs_dir, exist_ok=True)
    import torchvision.utils as vutils
    import cv2 as _cv2

    torch.cuda.empty_cache()
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for save_i, idx in enumerate(eval_indices):
        ts      = dataset.rgbinfo_dict['timestamp'][idx]
        nearest = int(np.argmin(np.abs(traj_times - ts)))
        if np.abs(traj_times[nearest] - ts) > 0.5:
            continue

        row         = traj[nearest]
        rot_mat     = R.from_quat(row[4:8]).as_matrix()
        c2w         = np.eye(4)
        c2w[:3, :3] = rot_mat
        c2w[:3, 3]  = row[1:4]
        w2c = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)

        # Load GT at original full resolution from disk
        rgb_path = dataset.rgbinfo_dict['filepath'][idx]
        rgb_raw  = _cv2.imread(rgb_path)
        gt = torch.tensor(rgb_raw[..., [2, 1, 0]], dtype=torch.float32
                          ).permute(2, 0, 1).to(device) / 255.0

        if gt.shape[1] != intr['H'] or gt.shape[2] != intr['W']:
            gt = F.interpolate(gt.unsqueeze(0),
                               size=(intr['H'], intr['W']),
                               mode='bilinear', align_corners=False).squeeze(0)

        with torch.no_grad():
            rendered = mapper.render(w2c, intr)['rgb'].clamp(0.0, 1.0)

        gt_b = gt.unsqueeze(0)
        rd_b = rendered.unsqueeze(0)

        psnr_vals.append(psnr(rd_b, gt_b).mean().item())
        ssim_vals.append(ssim(rd_b, gt_b).item())
        lpips_vals.append(lpips_fn(rd_b * 2 - 1, gt_b * 2 - 1).item())

        vutils.save_image(gt,       os.path.join(render_imgs_dir, f'{save_i:04d}_gt.png'))
        vutils.save_image(rendered, os.path.join(render_imgs_dir, f'{save_i:04d}_rendered.png'))

    if not psnr_vals:
        print(f"[render_eval] No valid frames evaluated for {seq_name}")
        return None

    result = {
        'psnr':  float(np.mean(psnr_vals)),
        'ssim':  float(np.mean(ssim_vals)),
        'lpips': float(np.mean(lpips_vals)),
        'n':     len(psnr_vals),
    }
    print(f"[render_eval] {seq_name}  PSNR={result['psnr']:.2f}  "
          f"SSIM={result['ssim']:.4f}  LPIPS={result['lpips']:.4f}  (n={result['n']})")
    return result


def run_sequence(seq_path, config_template_path, output_folder, run_slam=True, run_eval=True, run_plot=True, use_full_traj=False):
    """Run VINGS-Mono on a single sequence and evaluate."""
    seq_name = os.path.basename(seq_path)
    seq_output = os.path.join(output_folder, seq_name)
    os.makedirs(seq_output, exist_ok=True)
    print(f"Sequence output: {seq_output}")
    # Load and modify config for this sequence
    with open(config_template_path, 'r') as f:
        config = yaml.safe_load(f)

    config['dataset']['root'] = seq_path
    config['output']['save_dir'] = seq_output

    # Save modified config
    seq_config_path = os.path.join(seq_output, 'config.yaml')
    with open(seq_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run SLAM
    if run_slam:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        
        python_file = os.path.join(scripts_dir, 'run.py')
        cmd = f' python {python_file} {seq_config_path} > {seq_output}/log.txt 2>&1'
        print(f"Running: {cmd}")
        os.system(cmd)
    print(seq_config_path)
    # Find trajectory file
    traj_file = None

    # VINGS-Mono creates output in timestamped subfolder
    subdirs = sorted([d for d in glob(os.path.join(seq_output, '*')) if os.path.isdir(d)])

    if use_full_traj:
        # Prefer traj_full.txt (per-frame, from PoseTrajectoryFiller)
        for search_dir in [seq_output] + subdirs:
            candidate = os.path.join(search_dir, 'traj_full.txt')
            if os.path.exists(candidate):
                traj_file = candidate
                print(f"Using full trajectory: {traj_file}")
                break

    if traj_file is None:
        # Look for droid_c2w folder and combine keyframe poses
        rgb_dir = os.path.join(seq_path, 'rgb')
        for search_dir in [seq_output] + subdirs:
            droid_c2w_dir = os.path.join(search_dir, 'droid_c2w')
            if os.path.isdir(droid_c2w_dir):
                combined_traj = os.path.join(search_dir, 'traj_combined.txt')
                traj_file = combine_poses_to_tum(droid_c2w_dir, combined_traj, rgb_dir=rgb_dir)
                if traj_file:
                    print(f"Combined trajectory saved to: {traj_file}")
                    break

    # Fallback: check for existing trajectory files
    if traj_file is None:
        for search_dir in [seq_output] + subdirs:
            for traj_name in ['traj_full.txt', 'traj_combined.txt', 'traj_kf_beforeBA.txt', 'traj_kf_afterBA.txt']:
                candidate = os.path.join(search_dir, traj_name)
                if os.path.exists(candidate):
                    traj_file = candidate
                    break
            if traj_file:
                break

    result = {'seq': seq_name, 'ate': None, 'scale': None, 'traj_file': None, 'gt_file': None}

    # Evaluate with evo_ape
    if run_eval and traj_file:
        gt_file = os.path.join(seq_path, 'gt_imu.txt')
        traj_basename = os.path.basename(traj_file).replace('.txt', '')
        log_ape_file = os.path.join(seq_output, f'log_ape_{traj_basename}.txt')

        cmd = f'evo_ape tum -vas --no_warnings ' \
              f'--save_results {seq_output}/ape_results.zip ' \
              f'{gt_file} {traj_file} > {log_ape_file} 2>&1'
        print(f"Evaluating: {cmd}")
        os.system(cmd)

        # Parse results
        try:
            with open(log_ape_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if 'rmse' in line.lower():
                    result['ate'] = float(line.split()[-1])
                if line.lower().startswith('scale correction:'):
                    result['scale'] = float(line.split()[-1])
        except Exception as e:
            print(f"Error parsing results for {seq_name}: {e}")

        # Store paths for visualization
        result['traj_file'] = traj_file
        result['gt_file'] = gt_file

        # Generate trajectory plots
        if run_plot:
            try:
                est_traj = load_tum_trajectory(traj_file)
                gt_traj = load_tum_trajectory(gt_file)
                if len(est_traj) > 0 and len(gt_traj) > 0:
                    plot_trajectory_2d(est_traj, gt_traj, seq_name,
                                       os.path.join(seq_output, f'trajectory_2d_{seq_name}.png'))
                    plot_trajectory_3d(est_traj, gt_traj, seq_name,
                                       os.path.join(seq_output, f'trajectory_3d_{seq_name}.png'))
            except Exception as e:
                print(f"Error generating plots for {seq_name}: {e}")

    return result


def main():
    hostname = socket.gethostname()

    # Configuration
    if 'euler' in hostname or hostname.startswith('eu-'):
        dataset_base = '/cluster/project/cvg/zihzhu/Datasets/rpngar'
    else:
        dataset_base = '/home/zihzhu/data/Datasets/rpngar'

    config_template = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '../configs/rpng/rpngar_table.yaml')

    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '../output_noLoop_noMetric_MappingNoMask_fulltraj/rpngar_eval')

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run VINGS-Mono on RPNG-AR dataset')
    parser.add_argument('--output', type=str, default=output_folder, help='Output folder')
    parser.add_argument('--config', type=str, default=config_template, help='Config template')
    parser.add_argument('--dataset', type=str, default=dataset_base, help='Dataset base path')
    parser.add_argument('--seqs', type=str, nargs='+', default=None, help='Specific sequences to run (e.g., table_01 table_02)')
    parser.add_argument('--skip-slam',   action='store_true', help='Skip SLAM, only run evaluation')
    parser.add_argument('--skip-eval',   action='store_true', help='Skip evaluation')
    parser.add_argument('--no-plot',     action='store_true', help='Skip generating plots')
    parser.add_argument('--full-traj',   action='store_true',
                        help='Evaluate on traj_full.txt (per-frame, from PoseTrajectoryFiller) '
                             'instead of traj_combined.txt (keyframes only)')
    parser.add_argument('--render-eval', action='store_true',
                        help='Render from saved 2DGS model and compute PSNR/SSIM/LPIPS')
    parser.add_argument('--render-n',    type=int, default=50,
                        help='Number of random frames for render evaluation (default: 50)')
    args = parser.parse_args()

    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    # Get sequences
    if args.seqs:
        seqs = [os.path.join(args.dataset, s) for s in args.seqs]
    else:
        seqs = sorted(glob(os.path.join(args.dataset, 'table_*')))

    print(f"Output folder: {output_folder}")
    print(f"Sequences: {[os.path.basename(s) for s in seqs]}")

    # Run all sequences
    results = []
    for seq in seqs:
        if not os.path.isdir(seq):
            print(f"Skipping {seq} (not a directory)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {seq}")
        print(f"{'='*60}")

        result = run_sequence(
            seq_path=seq,
            config_template_path=args.config,
            output_folder=output_folder,
            run_slam=not args.skip_slam,
            run_eval=not args.skip_eval,
            run_plot=not args.no_plot,
            use_full_traj=args.full_traj,
        )

        if args.render_eval:
            render_result = render_and_eval_sequence(
                seq_path=seq,
                output_folder=output_folder,
                use_full_traj_for_poses=args.full_traj,
                n_eval=args.render_n,
            )
            result['render'] = render_result
        else:
            result['render'] = None

        results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    ate_values, scale_errors = [], []
    psnr_values, ssim_values, lpips_values = [], [], []

    has_render = any(r.get('render') is not None for r in results)
    hdr = f"{'Sequence':<15} {'ATE [cm]':<12} {'Scale [%]':<12}"
    if has_render:
        hdr += f" {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8}"
    print(hdr)
    print("-" * (len(hdr) + 5))

    for r in results:
        ate_cm    = r['ate'] * 100 if r['ate'] else None
        scale_err = abs(1 - r['scale']) * 100 if r['scale'] else None
        row = f"{r['seq']:<15} {f'{ate_cm:.2f}' if ate_cm is not None else 'N/A':<12} " \
              f"{f'{scale_err:.2f}' if scale_err is not None else 'N/A':<12}"
        if has_render:
            rr = r.get('render')
            psnr_s  = f"{rr['psnr']:.2f}"  if rr else 'N/A'
            ssim_s  = f"{rr['ssim']:.4f}"  if rr else 'N/A'
            lpips_s = f"{rr['lpips']:.4f}" if rr else 'N/A'
            row += f" {psnr_s:<8} {ssim_s:<8} {lpips_s:<8}"
        print(row)

        if ate_cm    is not None: ate_values.append(ate_cm)
        if scale_err is not None: scale_errors.append(scale_err)
        if r.get('render'):
            psnr_values.append(r['render']['psnr'])
            ssim_values.append(r['render']['ssim'])
            lpips_values.append(r['render']['lpips'])

    print("-" * (len(hdr) + 5))
    mean_row = f"{'Mean':<15} {np.mean(ate_values):<12.2f} {np.mean(scale_errors):<12.2f}" \
               if ate_values else ""
    if has_render and psnr_values:
        mean_row += (f" {np.mean(psnr_values):<8.2f}"
                     f" {np.mean(ssim_values):<8.3f}"
                     f" {np.mean(lpips_values):<8.3f}")
    if mean_row:
        print(mean_row)

    # LaTeX output
    if ate_values:
        print(f"\nLaTeX format:")
        print("ATE [cm] & " + " & ".join(f"{v:.2f}" for v in ate_values)
              + f" & {np.mean(ate_values):.2f} \\\\")
        print("Scale [\\%] & " + " & ".join(f"{v:.2f}" for v in scale_errors)
              + f" & {np.mean(scale_errors):.2f} \\\\")
    if psnr_values:
        print("PSNR & " + " & ".join(f"{v:.2f}" for v in psnr_values)
              + f" & {np.mean(psnr_values):.2f} \\\\")
        print("SSIM & " + " & ".join(f"{v:.3f}" for v in ssim_values)
              + f" & {np.mean(ssim_values):.3f} \\\\")
        print("LPIPS & " + " & ".join(f"{v:.3f}" for v in lpips_values)
              + f" & {np.mean(lpips_values):.3f} \\\\")

    # Generate summary plot
    if results and not args.no_plot:
        plot_summary(results, os.path.join(output_folder, 'evaluation_summary.png'))

    # Generate combined trajectory plot for all sequences
    if len(results) > 1 and not args.no_plot:
        try:
            fig, axes = plt.subplots(2, (len(results) + 1) // 2, figsize=(5 * ((len(results) + 1) // 2), 10))
            axes = axes.flatten() if len(results) > 2 else [axes] if len(results) == 1 else axes

            for idx, r in enumerate(results):
                if r['traj_file'] and r['gt_file'] and idx < len(axes):
                    try:
                        est_traj = load_tum_trajectory(r['traj_file'])
                        gt_traj = load_tum_trajectory(r['gt_file'])

                        aligned_est, aligned_gt = align_trajectories(est_traj, gt_traj)
                        if len(aligned_est) >= 3:
                            est_xyz = aligned_est[:, 1:4]
                            gt_xyz = aligned_gt[:, 1:4]
                            R_align, t_align, scale = umeyama_alignment(est_xyz, gt_xyz, with_scale=True)
                            est_aligned = (scale * (R_align @ est_xyz.T).T + t_align)

                            ax = axes[idx]
                            ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], 'b-', label='GT', linewidth=1.5)
                            ax.plot(est_aligned[:, 0], est_aligned[:, 1], 'r--', label='Est', linewidth=1.5)
                            ax.set_title(f"{r['seq']}\nATE: {r['ate']*100:.2f}cm" if r['ate'] else r['seq'])
                            ax.axis('equal')
                            ax.grid(True, alpha=0.3)
                            ax.legend(fontsize=8)
                    except Exception as e:
                        print(f"Error plotting {r['seq']}: {e}")

            # Hide unused axes
            for idx in range(len(results), len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle('All Trajectories (Top-Down View)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'all_trajectories.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved combined trajectory plot: {os.path.join(output_folder, 'all_trajectories.png')}")
        except Exception as e:
            print(f"Error generating combined plot: {e}")


if __name__ == "__main__":
    main()
