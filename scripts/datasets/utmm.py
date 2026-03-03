import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from lietorch import SE3
'''
UTMM Dataset structure:
datadir/
    ├── rgb/
    │   └── {timestamp_ns}.png
    ├── imu.txt   (timestamp_ns,gx,gy,gz,ax,ay,az)  comma-separated
    └── depth/    (optional)

Intrinsics and extrinsics are fixed across sequences (set in config).
'''

class UTMMDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_dir = cfg['dataset']['root']
        # c2i = T_imu_cam (already pre-inverted in config, just like rpngar)
        self.c2i = np.array(cfg['frontend']['c2i']) if cfg['frontend']['c2i'] is not None else np.eye(4)
        self.preload_rgbinfo()
        self.tqdm = tqdm(total=self.__len__())

    def __len__(self):
        return len(self.rgbinfo_dict['timestamp'])

    def preload_camtimestamp(self):
        return np.array(self.rgbinfo_dict['timestamp']).reshape(-1, 1)

    def preload_imu(self):
        '''
        Load imu.txt: timestamp_ns,gx,gy,gz,ax,ay,az  (comma-separated)
        Returns ndarray (N,7): [timestamp_s, gx_deg, gy_deg, gz_deg, ax, ay, az]
        '''
        imu_file = os.path.join(self.dataset_dir, 'imu_ours.txt')
        try:
            all_imu = np.loadtxt(imu_file, delimiter=',')
        except:
            all_imu = np.loadtxt(imu_file, delimiter=' ')
        # all_imu[:, 0] /= 1e9               # ns → s
        all_imu[:, 1:4] *= 180 / np.pi     # rad/s → deg/s (frontend convention)
        return all_imu

    def preload_rgbinfo(self):
        '''Load image paths and timestamps (filenames are ns timestamps).'''
        rgb_dir = os.path.join(self.dataset_dir, 'rgb_timestamp')
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        self.rgbinfo_dict = {
            'timestamp': [float(os.path.splitext(f)[0]) / 1e9 for f in rgb_files],
            'filepath':  [os.path.join(rgb_dir, f) for f in rgb_files],
        }

    def __getitem__(self, idx):
        resized_h = int(self.cfg['frontend']['image_size'][0])
        resized_w = int(self.cfg['frontend']['image_size'][1])

        rgb_raw = cv2.imread(self.rgbinfo_dict['filepath'][idx])

        # Undistort if distortion coefficients provided
        dist = self.cfg['intrinsic'].get('distortion_coeffs', None)
        if dist is not None:
            K = np.array([[self.cfg['intrinsic']['fu'], 0, self.cfg['intrinsic']['cu']],
                          [0, self.cfg['intrinsic']['fv'], self.cfg['intrinsic']['cv']],
                          [0, 0, 1]], dtype=np.float64)
            rgb_raw = cv2.undistort(rgb_raw, K, np.array(dist))

        rgb = (torch.tensor(cv2.resize(rgb_raw, (resized_w, resized_h)))[..., [2, 1, 0]]
               ).permute(2, 0, 1).unsqueeze(0).to(self.cfg['device']['tracker'])

        H = self.cfg['intrinsic']['H']
        W = self.cfg['intrinsic']['W']
        intrinsic = torch.tensor([
            self.cfg['intrinsic']['fu'] * (resized_w / W),
            self.cfg['intrinsic']['fv'] * (resized_h / H),
            self.cfg['intrinsic']['cu'] * (resized_w / W),
            self.cfg['intrinsic']['cv'] * (resized_h / H),
        ], dtype=torch.float32, device=self.cfg['device']['tracker'])

        self.tqdm.update(1)
        return {
            'timestamp': self.rgbinfo_dict['timestamp'][idx],
            'rgb':       rgb,
            'intrinsic': intrinsic,
        }

    def load_gt_dict(self):
        gt_file = os.path.join(self.dataset_dir, 'groundtruth.txt')
        if not os.path.exists(gt_file):
            return None
        with open(gt_file, 'r') as f:
            lines = [l.strip() for l in f if not l.startswith('#') and l.strip()]
        data = np.array([list(map(float, l.split())) for l in lines])
        timestamps = data[:, 0]
        poses = data[:, 1:]
        c2ws = SE3(torch.tensor(poses)).matrix().numpy()
        return {'timestamps': timestamps, 'c2ws': c2ws}


def get_dataset(config):
    return UTMMDataset(config)
