import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')
import numpy as np
import shutil
import torch
from lietorch import SE3
from frontend.dbaf import DBAFusion
from gaussian.gaussian_model import GaussianModel
from gaussian.vis_utils import save_ply, vis_map, vis_bev
import argparse
parser = argparse.ArgumentParser(description="Add config path.")
parser.add_argument("config")
parser.add_argument("--prefix", default='')
args = parser.parse_args()
config_path = args.config
from gaussian.general_utils import load_config, get_name
config = load_config(config_path)
import importlib
get_dataset = importlib.import_module(config["dataset"]["module"]).get_dataset
from vings_utils.middleware_utils import judge_and_package, retrieve_to_tracker, datapacket_to_nerfslam
from storage.storage_manage import StorageManager
from loop.loop_model import LoopModel
from metric.metric_model import Metric_Model
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation as ScipyR
if config['mode'] == 'vo_nerfslam': from frontend_vo.vio_slam import VioSLAM


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset  = get_dataset(cfg)
        cfg['frontend']['c2i'] = self.dataset.c2i # (4, 4), ndarray
        
        if self.cfg['mode'] == 'vio' or self.cfg['mode'] == 'vo':
            self.tracker = DBAFusion(cfg)
        elif self.cfg['mode'] == 'vo_nerfslam':     
            self.tracker = VioSLAM(cfg)
        else: assert False, "Error \"mode\" in config file."
        
        if 'phone' not in cfg['dataset']['module']: self.tracker.dataset_length = len(self.dataset)
        
        self.mapper = GaussianModel(cfg)
        
        self.looper = LoopModel(cfg)
        
        if 'use_metric' in cfg.keys() and cfg['use_metric']:
            self.metric_predictor = Metric_Model(cfg) 
        
        if 'use_storage_manager' in cfg.keys() and cfg['use_storage_manager']:
            self.use_storage_manager = True
            self.storage_manager = StorageManager(cfg)
            if cfg['dataset']['module'] != 'phone':
                self.storage_manager.dataset_length = self.dataset.rgbinfo_dict['timestamp'][-1] - self.dataset.rgbinfo_dict['timestamp'][0] 
        else:
            self.use_storage_manager = False

    def run(self):
        # Load imu data.
        self.tracker.frontend.all_imu   = self.dataset.preload_imu()
        self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
        
        mapper_run_times = 0
        
        # Run Tracking.
        for idx in tqdm(range(len(self.dataset))):
            
            data_packet = self.dataset[idx]
            
            if 'use_mobile' in self.cfg.keys() and self.cfg['use_mobile']:
                self.tracker.frontend.all_imu   = self.dataset.preload_imu()
                self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
            
            if 'use_metric' in self.cfg.keys() and self.cfg['use_metric']:
                if 'depth' not in data_packet.keys() or data_packet['depth'] is None:
                    data_packet['depth'] = self.metric_predictor.predict(data_packet['rgb'][0])
            
            self.tracker.frontend.all_imu   = self.dataset.preload_imu()
            self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
            # breakpoint()
            # torch.set_grad_enabled(False)
            self.tracker.track(data_packet if not self.cfg['mode']=='vo_nerfslam' else datapacket_to_nerfslam(data_packet, idx))
            # torch.set_grad_enabled(True)
            
            torch.cuda.empty_cache()
            # Judge whether new keyframe is added and package keyframe dict.
            viz_out = judge_and_package(self.tracker, data_packet['intrinsic'])
            
            if viz_out is not None:
                # Save and check.
                # Save all keyframe poses to droid_c2w/ for full trajectory
                for i, tstamp in enumerate(viz_out['viz_out_idx_to_f_idx']):
                    c2w = viz_out['poses'][i].cpu().numpy()
                    ts_sec = float(tstamp.item())  # timestamp in seconds
                    np.savetxt(f"{self.cfg['output']['save_dir']}/droid_c2w/{ts_sec:.6f}.txt", c2w)

                if self.cfg.get('use_mapper', True):
                    new_viz_out = self.mapper.run(viz_out, True)
                else:
                    new_viz_out = viz_out
                
                if 'use_loop' in list(self.cfg.keys()) and self.cfg['use_loop']:
                    if viz_out["global_kf_id"][-1] > 10 and viz_out["global_kf_id"][-1] % 3 == 0:
                        self.looper.run(self.mapper, self.tracker, viz_out, idx)

                if self.use_storage_manager and (idx+1) % 10 == 0:
                    self.storage_manager.run(self.tracker, self.mapper, viz_out)
                    torch.cuda.empty_cache()
                
                if self.cfg['use_vis'] and (idx+1) % 1 == 0:
                    if not self.cfg['use_storage_manager'] or self.storage_manager._xyz.shape[0]==0:
                        vis_map(self.tracker, self.mapper)
                        vis_bev(self.tracker, self.mapper) 
                    else:
                        self.storage_manager.vis_map_storage(self.tracker, self.mapper)    
                        self.storage_manager.vis_bev_storage(self.tracker, self.mapper)    
            
            if self.cfg.get('use_mapper', True) and (idx == len(self.dataset) - 1) and self.mapper._xyz.shape[0] > 0:
            # if ((idx+1) % 100 == 0 or (idx == len(self.dataset) - 1)) and self.mapper._xyz.shape[0] > 0:
                save_ply(self.mapper, idx, save_mode='2dgs')
                # save_ply(self.mapper, idx, save_mode='pth')

        # After tracking: fill in poses for all input frames (not just keyframes)
        self._save_full_trajectory()

    def _save_full_trajectory(self):
        """Fill per-frame poses for all dataset frames using SE3 Lie-algebra interpolation
        between keyframe poses from droid_c2w/.  This avoids the video-buffer rollup bug
        (video.poses[:N] only contains the last N keyframes after rollup, not the full
        trajectory).  droid_c2w/ snapshots are always correct regardless of rollup.

        Result saved as traj_full.txt in TUM format (one line per dataset frame).
        """
        import glob as globlib

        save_dir = self.cfg['output']['save_dir']
        droid_c2w_dir = os.path.join(save_dir, 'droid_c2w')
        pose_files = sorted(globlib.glob(os.path.join(droid_c2w_dir, '*.txt')))

        if not pose_files:
            print("[traj_filler] No keyframes in droid_c2w/, skipping full trajectory.")
            return

        try:
            print(f"[traj_filler] Loading {len(pose_files)} keyframe poses from droid_c2w/...")

            # Load keyframe c2w matrices → convert to w2c SE3 (7-dim TQ)
            kf_times, kf_w2c_tqs = [], []
            for pf in pose_files:
                ts = float(os.path.basename(pf)[:-4])   # filename = timestamp in seconds
                T_c2w = np.loadtxt(pf)
                if T_c2w.shape != (4, 4):
                    continue
                T_w2c = np.linalg.inv(T_c2w)
                q = ScipyR.from_matrix(T_w2c[:3, :3]).as_quat()  # qx qy qz qw
                t = T_w2c[:3, 3]
                kf_times.append(ts)
                kf_w2c_tqs.append(np.concatenate([t, q]))

            if not kf_times:
                print("[traj_filler] No valid keyframe poses found, skipping.")
                return

            # Sort keyframes by timestamp
            order = np.argsort(kf_times)
            kf_times   = np.array(kf_times)[order]
            kf_w2c_tqs = np.array(kf_w2c_tqs)[order]

            device = self.cfg['device']['tracker']
            ts_kf = torch.tensor(kf_times,   dtype=torch.float64, device=device)
            Ps    = SE3(torch.tensor(kf_w2c_tqs, dtype=torch.float64, device=device))
            N     = len(kf_times)

            # Collect all dataset frame timestamps (lightweight second pass)
            print("[traj_filler] Collecting dataset timestamps for interpolation...")
            all_tstamps = []
            for idx in tqdm(range(len(self.dataset)), desc='traj_filler', leave=False):
                data = self.dataset[idx]
                all_tstamps.append(data['timestamp'])

            n_images = len(all_tstamps)
            tt = torch.tensor(all_tstamps, dtype=torch.float64, device=device)

            # SE3 Lie-algebra interpolation
            t0_idx = torch.tensor(
                [(ts_kf <= t).sum().item() - 1 for t in tt],
                dtype=torch.long, device=device)
            t0_idx = torch.clamp(t0_idx, min=0)
            t1_idx = torch.where(t0_idx < N - 1, t0_idx + 1, t0_idx)

            dt      = ts_kf[t1_idx] - ts_kf[t0_idx] + 1e-6   # guard /0
            dP      = Ps[t1_idx] * Ps[t0_idx].inv()
            v       = dP.log() / dt.unsqueeze(-1)
            tau     = (tt - ts_kf[t0_idx]).unsqueeze(-1).clamp(min=0)  # no backward extrapolation
            full_w2c = SE3.exp(v * tau) * Ps[t0_idx]           # w2c interpolated poses

            c2ws = full_w2c.inv().matrix().cpu().numpy()        # (n_images, 4, 4) c2w

            # Save as TUM format: timestamp tx ty tz qx qy qz qw
            tum_data = []
            for i in range(n_images):
                T = c2ws[i]
                r = ScipyR.from_matrix(T[:3, :3])
                qx, qy, qz, qw = r.as_quat()
                tum_data.append([all_tstamps[i], T[0, 3], T[1, 3], T[2, 3], qx, qy, qz, qw])

            out_path = os.path.join(save_dir, 'traj_full.txt')
            np.savetxt(out_path, tum_data, fmt='%.9f')
            print(f"[traj_filler] Saved full trajectory ({n_images} frames, "
                  f"{N} keyframes): {out_path}")

        except Exception as exc:
            import traceback
            print(f"[traj_filler] ERROR: {exc}")
            traceback.print_exc()
            

if __name__ == '__main__':
    
    config['output']['save_dir'] = os.path.join(config['output']['save_dir'], get_name(config)+'-{}-'.format(config_path.split('/')[-1].strip('.yaml'))+args.prefix)
    os.makedirs(config['output']['save_dir']+'/droid_c2w', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/rgbdnua', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/ply', exist_ok=True)
    if 'debug_mode' in list(config.keys()) and config['debug_mode']:
        os.makedirs(config['output']['save_dir']+'/debug_dict', exist_ok=True)
    shutil.copy(config_path, config['output']['save_dir']+'/config.yaml')
    
    runner = Runner(config)
    torch.backends.cudnn.benchmark = True
    
    runner.run()
    
    