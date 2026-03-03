import os
import sys
import yaml
import numpy as np
from glob import glob
import argparse
import socket
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SEQUENCES = [
    'CBD_Building_01', 'CBD_Building_02',
    'HKU_Campus', 'Retail_Street', 'SYSU_01',
]

# Per-sequence fixed eval indices (random fallback when key absent)
FASTLIVO_EVAL_INDICES = {
    'CBD_Building_01': [838, 902, 1136, 521, 5, 571, 224, 549, 1181, 267, 108, 1039, 317, 1113, 724, 695, 760, 268, 8, 1079, 240, 14, 801, 795, 961, 986, 1012, 565, 641, 39, 923, 702, 596, 491, 124, 295, 1017, 1027, 472, 1163, 569, 545, 597, 1096, 1179, 52, 159, 619, 970, 665],
    'CBD_Building_02': [2010, 1655, 2264, 2036, 130, 2229, 224, 935, 581, 257, 118, 2213, 1408, 45, 1396, 628, 1555, 821, 546, 965, 713, 639, 793, 214, 311, 2058, 402, 560, 1707, 39, 452, 1002, 976, 910, 2074, 501, 1470, 1735, 2201, 1646, 1645, 1770, 971, 1472, 2182, 462, 1289, 1666, 1232, 1370],
    'HKU_Campus': [218, 389, 814, 328, 339, 200, 268, 764, 239, 883, 401, 74, 72, 696, 884, 449, 728, 116, 584, 177, 90, 123, 220, 330, 771, 212, 740, 867, 95, 56, 458, 715, 456, 84, 377, 292, 607, 871, 43, 385, 895, 109, 898, 20, 59, 512, 186, 96, 691, 539],
    'Retail_Street': [1071, 985, 408, 995, 108, 1069, 394, 1101, 594, 963, 532, 550, 485, 16, 685, 29, 536, 641, 1086, 952, 968, 55, 976, 1121, 329, 616, 542, 987, 15, 459, 440, 286, 836, 1327, 660, 123, 23, 1054, 527, 98, 334, 852, 888, 1329, 447, 1125, 520, 1024, 1055, 637],
    'SYSU_01': [112, 77, 864, 771, 151, 125, 116, 794, 509, 498, 59, 1331, 727, 358, 686, 147, 1038, 809, 46, 110, 516, 1135, 381, 1122, 798, 259, 438, 1144, 743, 694, 553, 45, 39, 1200, 585, 632, 135, 22, 75, 431, 167, 870, 49, 174, 1047, 717, 1034, 144, 1069, 201],
}


def psnr(img1, img2):
    import torch
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def _gaussian_window(window_size, sigma):
    import torch
    from math import exp as mexp
    gauss = torch.tensor([
        mexp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def ssim(img1, img2, window_size=11):
    import torch
    import torch.nn.functional as F
    channel = img1.size(1)
    _1d = _gaussian_window(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous().to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1  = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    s2  = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    s12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / (
               (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean()


def render_and_eval_sequence(seq_path, output_folder, use_full_traj_for_poses=True, n_eval=50):
    """Load saved 2DGS model, render at n_eval frames, compute PSNR/SSIM/LPIPS."""
    import importlib, torch, torch.nn.functional as F

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from gaussian.gaussian_model import GaussianModel
    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

    seq_name   = os.path.basename(seq_path)
    seq_output = os.path.join(output_folder, seq_name)
    subdirs    = sorted([d for d in glob(os.path.join(seq_output, '*')) if os.path.isdir(d)])

    # Locate run directory (non-empty ply/)
    run_dir = None
    for d in [seq_output] + subdirs:
        if glob(os.path.join(d, 'ply', '*_2dgs.ply')):
            run_dir = d
            break
    if run_dir is None:
        print(f'[render_eval] No ply/ directory for {seq_name}, skipping.')
        return None

    ply_files = sorted(glob(os.path.join(run_dir, 'ply', '*_2dgs.ply')))
    ply_path  = ply_files[-1]

    intrinsic_path = os.path.join(run_dir, 'ply', 'intrinsic.yaml')
    if not os.path.exists(intrinsic_path):
        print(f'[render_eval] No intrinsic.yaml for {seq_name}')
        return None

    # Locate trajectory
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
        print(f'[render_eval] No trajectory for {seq_name}')
        return None
    print(f'[render_eval] Using trajectory: {traj_file}')

    traj       = load_tum_trajectory(traj_file)
    traj_times = traj[:, 0]

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
    print(f'[render_eval] Rendering at full resolution: {intr["W"]}x{intr["H"]}')

    get_dataset_fn = importlib.import_module(cfg['dataset']['module']).get_dataset
    dataset  = get_dataset_fn(cfg)
    n_frames = len(dataset)

    if seq_name in FASTLIVO_EVAL_INDICES:
        eval_indices = [i for i in FASTLIVO_EVAL_INDICES[seq_name] if i < n_frames]
        print(f'[render_eval] Using fixed eval indices for {seq_name} (n={len(eval_indices)})')
    else:
        rng          = np.random.default_rng(42)
        eval_indices = sorted(rng.choice(n_frames, size=min(n_eval, n_frames), replace=False).tolist())

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
        print(f'[render_eval] No valid frames for {seq_name}')
        return None

    result = {
        'psnr':  float(np.mean(psnr_vals)),
        'ssim':  float(np.mean(ssim_vals)),
        'lpips': float(np.mean(lpips_vals)),
        'n':     len(psnr_vals),
    }
    print(f'[render_eval] {seq_name}  PSNR={result["psnr"]:.2f}  '
          f'SSIM={result["ssim"]:.4f}  LPIPS={result["lpips"]:.4f}  (n={result["n"]})')
    return result


def load_tum_trajectory(filepath):
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


def align_trajectories(est_traj, gt_traj, max_diff=0.05):
    aligned_est, aligned_gt = [], []
    gt_times = gt_traj[:, 0]
    for est in est_traj:
        idx = np.argmin(np.abs(gt_times - est[0]))
        if np.abs(gt_times[idx] - est[0]) < max_diff:
            aligned_est.append(est)
            aligned_gt.append(gt_traj[idx])
    return np.array(aligned_est), np.array(aligned_gt)


def umeyama_alignment(est_xyz, gt_xyz, with_scale=True):
    mu_est = est_xyz.mean(axis=0)
    mu_gt  = gt_xyz.mean(axis=0)
    est_c  = est_xyz - mu_est
    gt_c   = gt_xyz  - mu_gt
    H = est_c.T @ gt_c / len(est_xyz)
    U, S, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T
    scale = np.sum(S) / np.sum(est_c ** 2) * len(est_xyz) if with_scale else 1.0
    t_align = mu_gt - scale * R_align @ mu_est
    return R_align, t_align, scale


def matrix_to_tum(timestamp, T):
    tx, ty, tz = T[:3, 3]
    qx, qy, qz, qw = R.from_matrix(T[:3, :3]).as_quat()
    return [timestamp, tx, ty, tz, qx, qy, qz, qw]


def combine_poses_to_tum(droid_c2w_dir, output_file):
    pose_files = sorted(glob(os.path.join(droid_c2w_dir, '*.txt')))
    if not pose_files:
        return None
    tum_data = []
    for pf in pose_files:
        timestamp = float(os.path.basename(pf)[:-4])
        try:
            T = np.loadtxt(pf)
            if T.shape == (4, 4):
                tum_data.append(matrix_to_tum(timestamp, T))
        except Exception as e:
            print(f'Error loading {pf}: {e}')
    if not tum_data:
        return None
    tum_data = sorted(tum_data, key=lambda x: x[0])
    np.savetxt(output_file, tum_data, fmt='%.9f')
    return output_file


def plot_trajectory_2d(est_traj, gt_traj, title, output_path):
    aligned_est, aligned_gt = align_trajectories(est_traj, gt_traj)
    if len(aligned_est) < 3:
        print(f'Not enough aligned points for {title}')
        return
    est_xyz = aligned_est[:, 1:4]
    gt_xyz  = aligned_gt[:, 1:4]
    R_align, t_align, scale = umeyama_alignment(est_xyz, gt_xyz)
    est_aligned = (scale * (R_align @ est_xyz.T).T + t_align)
    ape = np.linalg.norm(est_aligned - gt_xyz, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], 'b-', label='GT', linewidth=2)
    ax.plot(est_aligned[:, 0], est_aligned[:, 1], 'r--', label='Est', linewidth=2)
    ax.scatter(gt_xyz[0, 0], gt_xyz[0, 1], c='g', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], c='m', s=100, marker='s', zorder=5, label='End')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_title('Top-Down (XY)')
    ax.legend(); ax.axis('equal'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'b-', label='GT', linewidth=2)
    ax.plot(est_aligned[:, 0], est_aligned[:, 2], 'r--', label='Est', linewidth=2)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Z [m]'); ax.set_title('Side View (XZ)')
    ax.legend(); ax.axis('equal'); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ts = aligned_est[:, 0] - aligned_est[0, 0]
    ax.plot(ts, ape * 100, 'r-', linewidth=1.5)
    ax.axhline(np.mean(ape) * 100, color='b', linestyle='--', label=f'Mean: {np.mean(ape)*100:.2f} cm')
    ax.axhline(np.sqrt(np.mean(ape**2)) * 100, color='g', linestyle='--', label=f'RMSE: {np.sqrt(np.mean(ape**2))*100:.2f} cm')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('APE [cm]'); ax.set_title('Absolute Position Error')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f'{title}  (scale={scale:.4f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def run_sequence(seq_path, config_template_path, output_folder, dataset_base,
                 run_slam=True, run_eval=True, run_plot=True, use_full_traj=False):
    seq_name   = os.path.basename(seq_path)
    seq_output = os.path.join(output_folder, seq_name)
    os.makedirs(seq_output, exist_ok=True)
    os.makedirs(os.path.join(seq_output, 'droid_c2w'), exist_ok=True)
    os.makedirs(os.path.join(seq_output, 'rgbdnua'),   exist_ok=True)
    os.makedirs(os.path.join(seq_output, 'ply'),        exist_ok=True)

    with open(config_template_path, 'r') as f:
        config = yaml.safe_load(f)
    config['dataset']['root']    = seq_path
    config['output']['save_dir'] = seq_output
    seq_config_path = os.path.join(seq_output, 'config.yaml')
    with open(seq_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    if run_slam:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        python_file = os.path.join(scripts_dir, 'run.py')
        cmd = f'python {python_file} {seq_config_path} > {seq_output}/log.txt 2>&1'
        print(f'Running: {cmd}')
        os.system(cmd)

    subdirs = sorted([d for d in glob(os.path.join(seq_output, '*')) if os.path.isdir(d)])

    # Prefer traj_full.txt when requested
    traj_file = None
    if use_full_traj:
        for search_dir in [seq_output] + subdirs:
            candidate = os.path.join(search_dir, 'traj_full.txt')
            if os.path.exists(candidate):
                traj_file = candidate
                print(f'Using full trajectory: {traj_file}')
                break

    # Fall back to keyframe poses
    if traj_file is None:
        for search_dir in [seq_output] + subdirs:
            droid_c2w_dir = os.path.join(search_dir, 'droid_c2w')
            if os.path.isdir(droid_c2w_dir) and glob(os.path.join(droid_c2w_dir, '*.txt')):
                combined = os.path.join(search_dir, 'traj_combined.txt')
                traj_file = combine_poses_to_tum(droid_c2w_dir, combined)
                if traj_file:
                    print(f'Combined trajectory: {traj_file}')
                    break

    result = {'seq': seq_name, 'ate': None, 'scale': None}

    if run_eval and traj_file:
        gt_file = os.path.join(dataset_base, 'pgt', f'{seq_name}.txt')
        if not os.path.exists(gt_file):
            print(f'GT not found: {gt_file}')
        else:
            traj_basename = os.path.basename(traj_file).replace('.txt', '')
            log_ape = os.path.join(seq_output, f'log_ape_{traj_basename}.txt')
            cmd = (f'evo_ape tum -vas --no_warnings '
                   f'--save_results {seq_output}/ape_results.zip '
                   f'{gt_file} {traj_file} > {log_ape} 2>&1')
            print(f'Evaluating: {cmd}')
            os.system(cmd)

            try:
                with open(log_ape) as f:
                    for line in f:
                        if 'rmse' in line.lower():
                            result['ate'] = float(line.split()[-1])
                        if line.lower().startswith('scale correction:'):
                            result['scale'] = float(line.split()[-1])
            except Exception as e:
                print(f'Error parsing results for {seq_name}: {e}')

            if run_plot and result['ate'] is not None:
                try:
                    est_traj = load_tum_trajectory(traj_file)
                    gt_traj  = load_tum_trajectory(gt_file)
                    if len(est_traj) > 0 and len(gt_traj) > 0:
                        plot_trajectory_2d(est_traj, gt_traj, seq_name,
                                           os.path.join(seq_output, f'trajectory_{seq_name}.png'))
                except Exception as e:
                    print(f'Error plotting {seq_name}: {e}')

    print(f'Done: {seq_name}  →  {seq_output}')
    return result


def main():
    hostname = socket.gethostname()
    if 'euler' in hostname or hostname.startswith('eu-'):
        dataset_base = '/cluster/project/cvg/zihzhu/Datasets/fast-livo2-dataset'
    else:
        dataset_base = '/home/zihzhu/data/Datasets/fast-livo2-dataset'

    config_template = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../configs/fastlivo/fastlivo.yaml'
    )
    output_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../output_noLoop_noMetric_MappingNoMask_fulltraj/fastlivo_eval'
    )

    parser = argparse.ArgumentParser(description='Run VINGS-Mono on FAST-LIVO2 dataset')
    parser.add_argument('--output',    type=str, default=output_folder)
    parser.add_argument('--config',    type=str, default=config_template)
    parser.add_argument('--dataset',   type=str, default=dataset_base)
    parser.add_argument('--seqs',      type=str, nargs='+', default=None,
                        help='Specific sequences, e.g. HKU_Campus CBD_Building_01')
    parser.add_argument('--skip-slam',   action='store_true')
    parser.add_argument('--skip-eval',   action='store_true')
    parser.add_argument('--no-plot',     action='store_true')
    parser.add_argument('--full-traj',   action='store_true',
                        help='Evaluate traj_full.txt (per-frame) instead of traj_combined.txt')
    parser.add_argument('--render-eval', action='store_true',
                        help='Render from saved 2DGS model and compute PSNR/SSIM/LPIPS')
    parser.add_argument('--render-n',    type=int, default=50,
                        help='Number of frames for render evaluation (default: 50)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.seqs:
        seqs = [os.path.join(args.dataset, s) for s in args.seqs]
    else:
        seqs = [os.path.join(args.dataset, s) for s in SEQUENCES
                if os.path.isdir(os.path.join(args.dataset, s))]

    print(f'Output: {args.output}')
    print(f'Sequences: {[os.path.basename(s) for s in seqs]}')

    results = []
    for seq in seqs:
        if not os.path.isdir(seq):
            print(f'Skipping {seq} (not found)')
            continue
        print(f'\n{"="*60}')
        print(f'Processing: {seq}')
        print(f'{"="*60}')
        result = run_sequence(
            seq_path=seq,
            config_template_path=args.config,
            output_folder=args.output,
            dataset_base=args.dataset,
            run_slam=not args.skip_slam,
            run_eval=not args.skip_eval,
            run_plot=not args.no_plot,
            use_full_traj=args.full_traj,
        )

        if args.render_eval:
            render_result = render_and_eval_sequence(
                seq_path=seq,
                output_folder=args.output,
                use_full_traj_for_poses=args.full_traj,
                n_eval=args.render_n,
            )
            result['render'] = render_result
        else:
            result['render'] = None

        results.append(result)

    # Summary
    print(f'\n{"="*60}')
    print('FAST-LIVO2 RESULTS SUMMARY')
    print(f'{"="*60}')
    has_render = any(r.get('render') is not None for r in results)
    hdr = f'{"Sequence":<20} {"ATE [cm]":<12} {"Scale [%]":<12}'
    if has_render:
        hdr += f' {"PSNR":<8} {"SSIM":<8} {"LPIPS":<8}'
    print(hdr)
    print('-' * len(hdr))
    ate_values, scale_errors = [], []
    psnr_values, ssim_values, lpips_values = [], [], []
    for r in results:
        ate_cm    = r['ate'] * 100 if r['ate'] is not None else None
        scale_err = abs(1 - r['scale']) * 100 if r['scale'] is not None else None
        row = (f"{r['seq']:<20} {f'{ate_cm:.2f}' if ate_cm is not None else 'N/A':<12} "
               f"{f'{scale_err:.2f}' if scale_err is not None else 'N/A':<12}")
        if has_render:
            rr = r.get('render')
            rr_psnr  = f"{rr['psnr']:.2f}"  if rr else 'N/A'
            rr_ssim  = f"{rr['ssim']:.3f}"  if rr else 'N/A'
            rr_lpips = f"{rr['lpips']:.3f}" if rr else 'N/A'
            row += f' {rr_psnr:<8} {rr_ssim:<8} {rr_lpips:<8}'
        print(row)
        if ate_cm    is not None: ate_values.append(ate_cm)
        if scale_err is not None: scale_errors.append(scale_err)
        if r.get('render'):
            psnr_values.append(r['render']['psnr'])
            ssim_values.append(r['render']['ssim'])
            lpips_values.append(r['render']['lpips'])
    print('-' * len(hdr))
    if ate_values:
        mean_row = f'{"Mean":<20} {np.mean(ate_values):<12.2f} {np.mean(scale_errors):<12.2f}'
        if has_render and psnr_values:
            mean_row += (f' {np.mean(psnr_values):<8.2f}'
                         f' {np.mean(ssim_values):<8.4f}'
                         f' {np.mean(lpips_values):<8.4f}')
        print(mean_row)
        print('\nLaTeX:')
        print('ATE [cm] & ' + ' & '.join(f'{v:.2f}' for v in ate_values) + f' & {np.mean(ate_values):.2f} \\\\')
        if scale_errors:
            print('Scale [%] & ' + ' & '.join(f'{v:.2f}' for v in scale_errors) + f' & {np.mean(scale_errors):.2f} \\\\')
    if psnr_values:
        print('PSNR & ' + ' & '.join(f'{v:.2f}' for v in psnr_values) + f' & {np.mean(psnr_values):.2f} \\\\')
        print('SSIM & ' + ' & '.join(f'{v:.4f}' for v in ssim_values) + f' & {np.mean(ssim_values):.4f} \\\\')
        print('LPIPS & ' + ' & '.join(f'{v:.4f}' for v in lpips_values) + f' & {np.mean(lpips_values):.4f} \\\\')


if __name__ == '__main__':
    main()
