import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from lietorch import SE3
import math
'''
RPNG-AR Dataset structure:
datadir/
    ├── rgb/
    │   └── {timestamp_ns}.png
    ├── imu.txt  (timestamp_ns,gx,gy,gz,ax,ay,az)
    └── gt_imu.txt (TUM format)
'''

class RPNGARDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_dir = cfg['dataset']['root']
        self.preload_rgbinfo()
        # c2i: camera to IMU transformation (4x4)
        self.c2i = np.array(cfg['frontend']['c2i']) if cfg['frontend']['c2i'] is not None else np.eye(4)
        self.tqdm = tqdm(total=self.__len__())

    def __len__(self):
        return len(self.rgbinfo_dict['timestamp'])

    def preload_camtimestamp(self):
        # Return camera timestamps in seconds
        return np.array(self.rgbinfo_dict['timestamp']).reshape(-1, 1)

    def preload_imu(self):
        '''
        Load IMU data from imu.txt
        Format: timestamp_ns,gx,gy,gz,ax,ay,az (comma-separated)
        Returns: ndarray of shape (N, 7) with [timestamp_s, gx, gy, gz, ax, ay, az]
        '''
        imu_file = os.path.join(self.dataset_dir, 'imu.txt')
        all_imu = np.loadtxt(imu_file, delimiter=',')
        # Convert nanoseconds to seconds
        all_imu[:, 0] = all_imu[:, 0] / 1e9
        all_imu[:,1:4] *= 180/math.pi
        return all_imu

    def preload_rgbinfo(self):
        '''
        Load RGB image paths and timestamps.
        Filenames are timestamps in nanoseconds.
        '''
        rgb_dir = os.path.join(self.dataset_dir, 'rgb')
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])

        rgbinfo_dict = {}
        # Extract timestamp from filename (in nanoseconds) and convert to seconds
        timestamps = [float(os.path.splitext(f)[0]) / 1e9 for f in rgb_files]
        rgbinfo_dict['timestamp'] = timestamps
        rgbinfo_dict['filepath'] = [os.path.join(rgb_dir, f) for f in rgb_files]
        self.rgbinfo_dict = rgbinfo_dict

    def __getitem__(self, idx):
        resized_h, resized_w = int(self.cfg['frontend']['image_size'][0]), int(self.cfg['frontend']['image_size'][1])
        rgb_raw = cv2.imread(self.rgbinfo_dict['filepath'][idx])

        # Undistort if distortion coefficients are provided
        if 'distortion_coeffs' in self.cfg['intrinsic'] and self.cfg['intrinsic']['distortion_coeffs'] is not None:
            K = np.eye(3)
            K[0, 0] = self.cfg['intrinsic']['fu']  # fx
            K[1, 1] = self.cfg['intrinsic']['fv']  # fy
            K[0, 2] = self.cfg['intrinsic']['cu']  # cx
            K[1, 2] = self.cfg['intrinsic']['cv']  # cy
            rgb_raw = cv2.undistort(rgb_raw, K, np.array(self.cfg['intrinsic']['distortion_coeffs']))


        rgb = (torch.tensor(cv2.resize(rgb_raw, (resized_w, resized_h)))[..., [2, 1, 0]]).permute(2, 0, 1).unsqueeze(0).to(self.cfg['device']['tracker'])

        # Scale factors: height_scale for vertical (y), width_scale for horizontal (x)
        height_scale = resized_h / self.cfg['intrinsic']['H']
        width_scale = resized_w / self.cfg['intrinsic']['W']
        # Intrinsics must be in order [fx, fy, cx, cy] as expected by projective_ops.py
        # fu=fx (horizontal focal), fv=fy (vertical focal), cu=cx (horizontal pp), cv=cy (vertical pp)
        intrinsic = torch.tensor([
            self.cfg['intrinsic']['fu'] * width_scale,   # fx scaled by width ratio
            self.cfg['intrinsic']['fv'] * height_scale,  # fy scaled by height ratio
            self.cfg['intrinsic']['cu'] * width_scale,   # cx scaled by width ratio
            self.cfg['intrinsic']['cv'] * height_scale   # cy scaled by height ratio
        ], dtype=torch.float32, device=self.cfg['device']['tracker'])

        data_packet = {}
        data_packet['timestamp'] = self.rgbinfo_dict['timestamp'][idx]
        data_packet['rgb'] = rgb
        data_packet['intrinsic'] = intrinsic

        self.tqdm.update(1)
        return data_packet

    def load_gt_dict(self):
        '''
        Load ground truth from gt_imu.txt (TUM format).
        Format: timestamp tx ty tz qx qy qz qw
        '''
        gt_file = os.path.join(self.dataset_dir, 'gt_imu.txt')
        # Skip comment lines
        with open(gt_file, 'r') as f:
            lines = [l.strip() for l in f if not l.startswith('#') and l.strip()]

        data = np.array([list(map(float, l.split())) for l in lines])
        timestamps = data[:, 0]
        # Convert to SE3 format: tx ty tz qx qy qz qw
        poses = data[:, 1:]  # [tx, ty, tz, qx, qy, qz, qw]
        c2ws = SE3(torch.tensor(poses)).matrix().numpy()

        gt_dict = {'timestamps': timestamps, 'c2ws': c2ws}
        return gt_dict

def get_dataset(config):
    return RPNGARDataset(config)
