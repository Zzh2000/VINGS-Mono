import os
import sys
import yaml
import numpy as np
from glob import glob
import socket
import shutil
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


# All EuRoC sequences
EUROC_SEQUENCES = [
    'MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult', 'MH_05_difficult',
    'V1_01_easy', 'V1_02_medium', 'V1_03_difficult',
    'V2_01_easy', 'V2_02_medium', 'V2_03_difficult',
]


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


def euroc_gt_to_tum(gt_csv_path, tum_output_path):
    """Convert EuRoC GT CSV to TUM format.

    EuRoC GT format: #timestamp[ns], px, py, pz, qw, qx, qy, qz, vx, vy, vz, ...
    TUM format:       timestamp tx ty tz qx qy qz qw
    """
    if not os.path.exists(gt_csv_path):
        return None
    data = np.loadtxt(gt_csv_path, delimiter=',')
    # columns: 0=ts_ns, 1=px, 2=py, 3=pz, 4=qw, 5=qx, 6=qy, 7=qz, ...
    timestamps = data[:, 0] / 1e9
    px, py, pz = data[:, 1], data[:, 2], data[:, 3]
    qw, qx, qy, qz = data[:, 4], data[:, 5], data[:, 6], data[:, 7]
    tum_data = np.column_stack([timestamps, px, py, pz, qx, qy, qz, qw])
    np.savetxt(tum_output_path, tum_data, fmt='%.9f')
    return tum_output_path


def matrix_to_tum(timestamp, T):
    """Convert 4x4 transformation matrix to TUM format."""
    tx, ty, tz = T[:3, 3]
    rot = R.from_matrix(T[:3, :3])
    qx, qy, qz, qw = rot.as_quat()  # scipy returns [x, y, z, w]
    return [timestamp, tx, ty, tz, qx, qy, qz, qw]


def combine_poses_to_tum(droid_c2w_dir, output_file):
    """Combine per-keyframe pose .txt files into a single TUM trajectory."""
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
            print(f"Error loading {pf}: {e}")

    if not tum_data:
        return None

    tum_data = sorted(tum_data, key=lambda x: x[0])
    np.savetxt(output_file, tum_data, fmt='%.9f')
    return output_file


def align_trajectories(est_traj, gt_traj, max_diff=0.01):
    """Align estimated trajectory to ground truth by nearest timestamp."""
    aligned_est, aligned_gt = [], []
    gt_times = gt_traj[:, 0]
    for est in est_traj:
        idx = np.argmin(np.abs(gt_times - est[0]))
        if np.abs(gt_times[idx] - est[0]) < max_diff:
            aligned_est.append(est)
            aligned_gt.append(gt_traj[idx])
    return np.array(aligned_est), np.array(aligned_gt)


def umeyama_alignment(est_xyz, gt_xyz, with_scale=True):
    """Align est to gt using Umeyama's method. Returns R, t, s."""
    mu_est = est_xyz.mean(axis=0)
    mu_gt = gt_xyz.mean(axis=0)
    est_c = est_xyz - mu_est
    gt_c = gt_xyz - mu_gt
    H = est_c.T @ gt_c / len(est_xyz)
    U, S, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T
    scale = np.sum(S) / np.sum(est_c ** 2) * len(est_xyz) if with_scale else 1.0
    t_align = mu_gt - scale * R_align @ mu_est
    return R_align, t_align, scale


def plot_trajectory_2d(est_traj, gt_traj, title, output_path):
    """Plot 2D trajectory comparison."""
    aligned_est, aligned_gt = align_trajectories(est_traj, gt_traj)
    if len(aligned_est) < 3:
        print(f"Not enough aligned points for {title}")
        return

    est_xyz = aligned_est[:, 1:4]
    gt_xyz = aligned_gt[:, 1:4]
    R_align, t_align, scale = umeyama_alignment(est_xyz, gt_xyz, with_scale=True)
    est_aligned = (scale * (R_align @ est_xyz.T).T + t_align)
    ape = np.linalg.norm(est_aligned - gt_xyz, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(est_aligned[:, 0], est_aligned[:, 1], 'r--', label='Estimated', linewidth=2)
    ax.scatter(gt_xyz[0, 0], gt_xyz[0, 1], c='g', s=100, marker='o', zorder=5, label='Start')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_title('Top-Down (XY)')
    ax.legend(); ax.axis('equal'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(est_aligned[:, 0], est_aligned[:, 2], 'r--', label='Estimated', linewidth=2)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Z [m]'); ax.set_title('Side (XZ)')
    ax.legend(); ax.axis('equal'); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ts = aligned_est[:, 0] - aligned_est[0, 0]
    ax.plot(ts, ape * 100, 'r-', linewidth=1.5)
    ax.axhline(y=np.mean(ape) * 100, color='b', linestyle='--',
               label=f'Mean: {np.mean(ape)*100:.2f} cm')
    ax.axhline(y=np.sqrt(np.mean(ape**2)) * 100, color='g', linestyle='--',
               label=f'RMSE: {np.sqrt(np.mean(ape**2))*100:.2f} cm')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('APE [cm]'); ax.set_title('APE over time')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f'{title}\nScale: {scale:.4f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plot: {output_path}")


def run_sequence(seq_path, config_template_path, output_folder, run_slam=True, run_eval=True, run_plot=True):
    """Run VINGS-Mono on a single EuRoC sequence and evaluate."""
    seq_name = os.path.basename(seq_path)
    seq_output = os.path.join(output_folder, seq_name)
    os.makedirs(seq_output, exist_ok=True)

    with open(config_template_path, 'r') as f:
        config = yaml.safe_load(f)

    config['dataset']['root'] = seq_path
    config['output']['save_dir'] = seq_output

    seq_config_path = os.path.join(seq_output, 'config.yaml')
    with open(seq_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    if run_slam:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        python_file = os.path.join(scripts_dir, 'run.py')
        cmd = f'python {python_file} {seq_config_path} > {seq_output}/log.txt 2>&1'
        print(f"Running: {cmd}")
        os.system(cmd)

    # Locate trajectory output
    traj_file = None
    subdirs = sorted([d for d in glob(os.path.join(seq_output, '*')) if os.path.isdir(d)])
    for search_dir in [seq_output] + subdirs:
        droid_c2w_dir = os.path.join(search_dir, 'droid_c2w')
        if os.path.isdir(droid_c2w_dir):
            combined_traj = os.path.join(search_dir, 'traj_combined.txt')
            traj_file = combine_poses_to_tum(droid_c2w_dir, combined_traj)
            if traj_file:
                print(f"Combined trajectory: {traj_file}")
                break

    result = {'seq': seq_name, 'ate': None, 'scale': None, 'traj_file': None, 'gt_file': None}

    # Convert EuRoC GT to TUM format
    # gt_csv = os.path.join(seq_path, 'mav0', 'state_groundtruth_estimate0', 'data.csv')
    # gt_tum = os.path.join(seq_output, 'gt_tum.txt')
    # gt_file = euroc_gt_to_tum(gt_csv, gt_tum)
    print(seq_name)
    gt_file = f'/cluster/project/cvg/zihzhu/VIGS-SLAM-release-prev/euroc_groundtruth/{seq_name}_sec.txt'
    if run_eval and traj_file and gt_file:
        traj_basename = os.path.basename(traj_file).replace('.txt', '')
        log_ape_file = os.path.join(seq_output, f'log_ape_{traj_basename}.txt')

        cmd = (f'evo_ape tum -vas --no_warnings '
               f'--save_results {seq_output}/ape_results.zip '
               f'{gt_file} {traj_file} > {log_ape_file} 2>&1')
        print(f"Evaluating: {cmd}")
        os.system(cmd)

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

        result['traj_file'] = traj_file
        result['gt_file'] = gt_file

        if run_plot and result['ate'] is not None:
            try:
                est_traj = load_tum_trajectory(traj_file)
                gt_traj = load_tum_trajectory(gt_file)
                if len(est_traj) > 0 and len(gt_traj) > 0:
                    plot_trajectory_2d(est_traj, gt_traj, seq_name,
                                       os.path.join(seq_output, f'trajectory_{seq_name}.png'))
            except Exception as e:
                print(f"Error generating plot for {seq_name}: {e}")

    return result


def plot_summary(results, output_path):
    valid = [r for r in results if r['ate'] is not None]
    if not valid:
        return
    seq_names = [r['seq'] for r in valid]
    ate_values = [r['ate'] * 100 for r in valid]
    scale_errors = [abs(1 - r['scale']) * 100 if r['scale'] else 0 for r in valid]

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(valid) * 1.2), 5))
    x = np.arange(len(seq_names))
    width = 0.6

    ax = axes[0]
    bars = ax.bar(x, ate_values, width, color='steelblue', edgecolor='black')
    ax.axhline(y=np.mean(ate_values), color='r', linestyle='--',
               label=f'Mean: {np.mean(ate_values):.2f} cm')
    for bar, val in zip(bars, ate_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Sequence'); ax.set_ylabel('ATE RMSE [cm]')
    ax.set_title('EuRoC — Absolute Trajectory Error')
    ax.set_xticks(x); ax.set_xticklabels(seq_names, rotation=45, ha='right')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    bars = ax.bar(x, scale_errors, width, color='coral', edgecolor='black')
    ax.axhline(y=np.mean(scale_errors), color='r', linestyle='--',
               label=f'Mean: {np.mean(scale_errors):.2f}%')
    for bar, val in zip(bars, scale_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Sequence'); ax.set_ylabel('Scale Error [%]')
    ax.set_title('EuRoC — Scale Error')
    ax.set_xticks(x); ax.set_xticklabels(seq_names, rotation=45, ha='right')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot: {output_path}")


def main():
    hostname = socket.gethostname()

    if 'euler' in hostname or hostname.startswith('eu-'):
        dataset_base = '/cluster/project/cvg/zihzhu/Datasets/euroc'
    else:
        dataset_base = '/home/zihzhu/data/Datasets/euroc'

    config_template = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '../configs/euroc/euroc.yaml')
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '../output_noLoop_noMetric/euroc_eval')

    import argparse
    parser = argparse.ArgumentParser(description='Run VINGS-Mono on EuRoC dataset')
    parser.add_argument('--output',  type=str, default=output_folder,  help='Output folder')
    parser.add_argument('--config',  type=str, default=config_template, help='Config template')
    parser.add_argument('--dataset', type=str, default=dataset_base,    help='Dataset base path')
    parser.add_argument('--seqs', type=str, nargs='+', default=None,
                        help='Specific sequences (e.g. MH_01_easy V1_01_easy). Default: all.')
    parser.add_argument('--skip-slam', action='store_true', help='Skip SLAM, only evaluate')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation')
    parser.add_argument('--no-plot',   action='store_true', help='Skip plots')
    args = parser.parse_args()

    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    seq_names = args.seqs if args.seqs else EUROC_SEQUENCES
    seqs = [os.path.join(args.dataset, s) for s in seq_names]

    print(f"Config:     {args.config}")
    print(f"Dataset:    {args.dataset}")
    print(f"Output:     {output_folder}")
    print(f"Sequences:  {seq_names}")

    results = []
    for seq in seqs:
        if not os.path.isdir(seq):
            print(f"Skipping {seq} (not found)")
            continue
        print(f"\n{'='*60}\nProcessing: {seq}\n{'='*60}")
        result = run_sequence(
            seq_path=seq,
            config_template_path=args.config,
            output_folder=output_folder,
            run_slam=not args.skip_slam,
            run_eval=not args.skip_eval,
            run_plot=not args.no_plot,
        )
        results.append(result)

    # Print summary table
    print(f"\n{'='*60}\nRESULTS SUMMARY\n{'='*60}")
    print(f"{'Sequence':<20} {'ATE [cm]':<12} {'Scale Error [%]':<15}")
    print("-" * 50)

    ate_values, scale_errors = [], []
    for r in results:
        ate_cm   = r['ate'] * 100 if r['ate'] is not None else None
        scale_err = abs(1 - r['scale']) * 100 if r['scale'] is not None else None
        print(f"{r['seq']:<20} {f'{ate_cm:.2f}' if ate_cm is not None else 'N/A':<12} "
              f"{f'{scale_err:.2f}' if scale_err is not None else 'N/A':<15}")
        if ate_cm is not None:
            ate_values.append(ate_cm)
        if scale_err is not None:
            scale_errors.append(scale_err)

    print("-" * 50)
    if ate_values:
        print(f"{'Mean':<20} {np.mean(ate_values):<12.2f} {np.mean(scale_errors):<15.2f}")

    # LaTeX table row
    if ate_values:
        print(f"\nLaTeX:")
        print("ATE [cm] & " + " & ".join(f"{v:.2f}" for v in ate_values)
              + f" & {np.mean(ate_values):.2f} \\\\")
        print("Scale [\\%] & " + " & ".join(f"{v:.2f}" for v in scale_errors)
              + f" & {np.mean(scale_errors):.2f} \\\\")

    if results and not args.no_plot:
        plot_summary(results, os.path.join(output_folder, 'evaluation_summary.png'))


if __name__ == "__main__":
    main()
