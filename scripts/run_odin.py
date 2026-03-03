import os
import sys
import yaml
import numpy as np
from glob import glob
import argparse
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

SEQUENCES = [
    'Basement1', 'Basement2', 'Basement3', 'Basement4',
    'Ferrari1',
    'Motorworld1', 'Motorworld2', 'Motorworld4', 'Motorworld5',
    'NTU_Campus1', 'NTU_Campus2',
    'NTU_Corridor1', 'NTU_Corridor2',
    'NTU_Office',
]


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
    ax.axhline(np.mean(ape) * 100, color='b', linestyle='--',
               label=f'Mean: {np.mean(ape)*100:.2f} cm')
    ax.axhline(np.sqrt(np.mean(ape**2)) * 100, color='g', linestyle='--',
               label=f'RMSE: {np.sqrt(np.mean(ape**2))*100:.2f} cm')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('APE [cm]'); ax.set_title('Absolute Position Error')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f'{title}  (scale={scale:.4f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def run_sequence(seq_path, config_template_path, output_folder,
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

    traj_file = None
    if use_full_traj:
        for search_dir in [seq_output] + subdirs:
            candidate = os.path.join(search_dir, 'traj_full.txt')
            if os.path.exists(candidate):
                traj_file = candidate
                print(f'Using full trajectory: {traj_file}')
                break

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
        gt_file = os.path.join(seq_path, 'MT-Pose.txt')
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
    dataset_base = '/cluster/scratch/zihzhu/odin'

    config_template = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../configs/odin/odin.yaml'
    )
    output_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../output_noLoop_noMetric/odin_eval'
    )

    parser = argparse.ArgumentParser(description='Run VINGS-Mono on ODIN dataset')
    parser.add_argument('--output',    type=str, default=output_folder)
    parser.add_argument('--config',    type=str, default=config_template)
    parser.add_argument('--dataset',   type=str, default=dataset_base)
    parser.add_argument('--seqs',      type=str, nargs='+', default=None,
                        help='Specific sequences, e.g. Basement1 Ferrari1')
    parser.add_argument('--skip-slam',  action='store_true')
    parser.add_argument('--skip-eval',  action='store_true')
    parser.add_argument('--no-plot',    action='store_true')
    parser.add_argument('--full-traj',  action='store_true',
                        help='Evaluate traj_full.txt (per-frame) instead of traj_combined.txt (keyframes)')
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
            run_slam=not args.skip_slam,
            run_eval=not args.skip_eval,
            run_plot=not args.no_plot,
            use_full_traj=args.full_traj,
        )
        results.append(result)

    # Summary
    print(f'\n{"="*60}')
    print('ODIN RESULTS SUMMARY')
    print(f'{"="*60}')
    hdr = f'{"Sequence":<20} {"ATE [cm]":<12} {"Scale [%]":<12}'
    print(hdr)
    print('-' * len(hdr))
    ate_values, scale_errors = [], []
    for r in results:
        ate_cm    = r['ate'] * 100 if r['ate'] is not None else None
        scale_err = abs(1 - r['scale']) * 100 if r['scale'] is not None else None
        print(f"{r['seq']:<20} {f'{ate_cm:.2f}' if ate_cm is not None else 'N/A':<12} "
              f"{f'{scale_err:.2f}' if scale_err is not None else 'N/A':<12}")
        if ate_cm    is not None: ate_values.append(ate_cm)
        if scale_err is not None: scale_errors.append(scale_err)
    print('-' * len(hdr))
    if ate_values:
        print(f'{"Mean":<20} {np.mean(ate_values):<12.2f} {np.mean(scale_errors):<12.2f}')
        print('\nLaTeX:')
        print('ATE [cm] & ' + ' & '.join(f'{v:.2f}' for v in ate_values)
              + f' & {np.mean(ate_values):.2f} \\\\')
        if scale_errors:
            print('Scale [%] & ' + ' & '.join(f'{v:.2f}' for v in scale_errors)
                  + f' & {np.mean(scale_errors):.2f} \\\\')


if __name__ == '__main__':
    main()
