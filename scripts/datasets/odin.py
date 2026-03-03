import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from lietorch import SE3
'''
ODIN Dataset structure:
datadir/
    ├── images/
    │   └── {timestamp_s}.jpg       (filename is already seconds)
    ├── imu.txt   (# header, timestamp_s,gx,gy,gz,ax,ay,az  comma-separated, gyro rad/s)
    ├── calib.txt (fx fy cx cy  space-separated)
    └── MT-Pose.txt  (TUM format: timestamp tx ty tz qx qy qz qw)

No distortion. c2i (T_imu_cam) is fixed in config frontend.c2i.
'''

class OdinDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_dir = cfg['dataset']['root']
        self._load_intrinsics()
        self.c2i = np.array(cfg['frontend']['c2i']) if cfg['frontend']['c2i'] is not None else np.eye(4)
        self.preload_rgbinfo()
        self.tqdm = tqdm(total=self.__len__())

    def _load_intrinsics(self):
        '''Read calib.txt: fx fy cx cy (space-separated). H/W from first image.'''
        intr = np.loadtxt(os.path.join(self.dataset_dir, 'calib.txt'))
        img_dir = os.path.join(self.dataset_dir, 'images')
        sample = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))[0]
        img = cv2.imread(os.path.join(img_dir, sample))
        H, W = img.shape[:2]
        self.cfg['intrinsic']['H']  = H
        self.cfg['intrinsic']['W']  = W
        self.cfg['intrinsic']['fu'] = float(intr[0])
        self.cfg['intrinsic']['fv'] = float(intr[1])
        self.cfg['intrinsic']['cu'] = float(intr[2])
        self.cfg['intrinsic']['cv'] = float(intr[3])
        self.cfg['intrinsic']['distortion_coeffs'] = None

    def __len__(self):
        return len(self.rgbinfo_dict['timestamp'])

    def preload_camtimestamp(self):
        return np.array(self.rgbinfo_dict['timestamp']).reshape(-1, 1)

    def preload_imu(self):
        '''
        Load imu.txt: timestamp_s,gx,gy,gz,ax,ay,az  (comma-separated, header with #)
        Returns ndarray (N,7): [timestamp_s, gx_deg, gy_deg, gz_deg, ax, ay, az]
        '''
        imu_file = os.path.join(self.dataset_dir, 'imu.txt')
        all_imu = np.loadtxt(imu_file, delimiter=',', comments='#')
        all_imu[:, 1:4] *= 180 / np.pi     # rad/s → deg/s (frontend convention)
        return all_imu

    def preload_rgbinfo(self):
        '''Load image paths and timestamps (filenames already in seconds).'''
        img_dir = os.path.join(self.dataset_dir, 'images')
        img_files = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))
        self.rgbinfo_dict = {
            'timestamp': [float(os.path.splitext(f)[0]) for f in img_files],
            'filepath':  [os.path.join(img_dir, f) for f in img_files],
        }

    def __getitem__(self, idx):
        resized_h = int(self.cfg['frontend']['image_size'][0])
        resized_w = int(self.cfg['frontend']['image_size'][1])

        rgb_raw = cv2.imread(self.rgbinfo_dict['filepath'][idx])

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
        gt_file = os.path.join(self.dataset_dir, 'MT-Pose.txt')
        if not os.path.exists(gt_file):
            return None
        with open(gt_file, 'r') as f:
            lines = [l.strip() for l in f if not l.startswith('#') and l.strip()]
        data = np.array([list(map(float, l.split())) for l in lines])
        timestamps = data[:, 0]
        poses = data[:, 1:]  # tx ty tz qx qy qz qw
        c2ws = SE3(torch.tensor(poses)).matrix().numpy()
        return {'timestamps': timestamps, 'c2ws': c2ws}


def get_dataset(config):
    return OdinDataset(config)
