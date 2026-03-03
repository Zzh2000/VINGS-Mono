import os
import numpy as np
import torch
import cv2
import yaml
from tqdm import tqdm
from lietorch import SE3

'''
EuRoC Dataset structure:
SeqName/mav0/
    ├── cam0/
    │   ├── data/          {timestamp_ns}.png
    │   ├── data.csv       #timestamp[ns],filename
    │   └── sensor.yaml    (T_BS = T_imu_cam, intrinsics, distortion)
    ├── imu0/
    │   └── data.csv       #timestamp[ns],gx,gy,gz,ax,ay,az  (rad/s, m/s^2)
    └── state_groundtruth_estimate0/
        └── data.csv       #timestamp[ns],px,py,pz,qw,qx,qy,qz,vx,vy,vz,...

Intrinsics and extrinsics are loaded from cam0/sensor.yaml.
'''


class EuRoCDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_dir = cfg['dataset']['root']
        mav0 = os.path.join(self.dataset_dir, 'mav0')
        self.mav0 = mav0

        # Load T_BS from cam0/sensor.yaml — T_BS = T_body_sensor = T_imu_cam
        # Used directly as c2i (no inversion needed)
        sensor_yaml = os.path.join(mav0, 'cam0', 'sensor.yaml')
        with open(sensor_yaml, 'r') as f:
            sensor = yaml.safe_load(f)
        t_bs_data = sensor['T_BS']['data']
        self.c2i = np.array(t_bs_data).reshape(4, 4)

        stride = cfg['dataset'].get('stride', 2)
        self.preload_rgbinfo(mav0, stride)
        self.tqdm = tqdm(total=self.__len__())

    def __len__(self):
        return len(self.rgbinfo_dict['timestamp'])

    def preload_camtimestamp(self):
        return np.array(self.rgbinfo_dict['timestamp']).reshape(-1, 1)

    def preload_imu(self):
        '''
        Load imu0/data.csv: #timestamp[ns],gx,gy,gz,ax,ay,az
        Returns ndarray (N,7): [timestamp_s, gx_deg, gy_deg, gz_deg, ax, ay, az]
        '''
        imu_file = os.path.join(self.mav0, 'imu0', 'data.csv')
        all_imu = np.loadtxt(imu_file, delimiter=',')
        all_imu[:, 0] /= 1e9            # ns → s
        all_imu[:, 1:4] *= 180 / np.pi  # rad/s → deg/s (frontend converts back)
        return all_imu

    def preload_rgbinfo(self, mav0, stride):
        '''Load image paths and timestamps from cam0/data.csv (stride applied).'''
        csv_file = os.path.join(mav0, 'cam0', 'data.csv')
        img_dir = os.path.join(mav0, 'cam0', 'data')

        # np.loadtxt skips '#' comment lines by default
        stamps_and_files = np.loadtxt(csv_file, dtype=str, delimiter=',')
        stamps_and_files = stamps_and_files[::stride]

        self.rgbinfo_dict = {
            'timestamp': [float(row[0]) / 1e9 for row in stamps_and_files],
            'filepath':  [os.path.join(img_dir, row[1]) for row in stamps_and_files],
        }

    def __getitem__(self, idx):
        resized_h = int(self.cfg['frontend']['image_size'][0])
        resized_w = int(self.cfg['frontend']['image_size'][1])

        rgb_raw = cv2.imread(self.rgbinfo_dict['filepath'][idx])

        # Undistort (EuRoC cam0 has significant radtan distortion)
        dist = self.cfg['intrinsic'].get('distortion_coeffs', None)
        if dist is not None:
            H0 = self.cfg['intrinsic']['H']
            W0 = self.cfg['intrinsic']['W']
            K = np.array([[self.cfg['intrinsic']['fu'], 0, self.cfg['intrinsic']['cu']],
                          [0, self.cfg['intrinsic']['fv'], self.cfg['intrinsic']['cv']],
                          [0, 0, 1]], dtype=np.float64)
            rgb_raw = cv2.undistort(rgb_raw, K, np.array(dist, dtype=np.float64))

        # Optional CLAHE contrast enhancement (EuRoC grayscale images benefit from this)
        if self.cfg['dataset'].get('use_clahe', False):
            clahe = cv2.createCLAHE(2.0, tileGridSize=(8, 8))
            gray = rgb_raw[:, :, 0]  # first channel (grayscale image loaded as BGR)
            gray_eq = clahe.apply(gray)
            rgb_raw = np.stack([gray_eq, gray_eq, gray_eq], axis=-1)

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
        gt_file = os.path.join(self.mav0, 'state_groundtruth_estimate0', 'data.csv')
        if not os.path.exists(gt_file):
            return None
        # EuRoC GT: #timestamp[ns], px, py, pz, qw, qx, qy, qz, vx, vy, vz, ...
        data = np.loadtxt(gt_file, delimiter=',')
        timestamps = data[:, 0] / 1e9
        # SE3 expects [tx, ty, tz, qx, qy, qz, qw]
        poses = np.column_stack([
            data[:, 1:4],   # tx, ty, tz
            data[:, 5:8],   # qx, qy, qz
            data[:, 4:5],   # qw
        ])
        c2ws = SE3(torch.tensor(poses, dtype=torch.float64)).matrix().numpy()
        return {'timestamps': timestamps, 'c2ws': c2ws}


def get_dataset(config):
    return EuRoCDataset(config)
