#!/bin/bash
# Setup script for VINGS-Mono on RTX 5090 (sm_120, CUDA 12.8)
# Usage: conda activate vings_vio && bash set_env_5090.sh
# All changes are local to this project folder.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing missing Python packages ==="
pip install mmengine==0.10.5
pip install lpips==0.1.4
pip install kornia==0.7.3 kornia-moons==0.2.9
pip install HTML4Vision==0.4.3
pip install psutil
pip install timm==0.6.7
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

echo "=== Building diff-surfel-rasterization (local) ==="
cd "$SCRIPT_DIR/submodules/diff-surfel-rasterization"
pip install . --no-build-isolation

echo "=== Building dbaf (droid_backends + lietorch) with sm_120 ==="
cd "$SCRIPT_DIR/submodules/dbaf"
python setup.py build_ext --parallel 4 install

echo "=== Creating mono_utils symlink (if missing) ==="
MONO_DIR="$SCRIPT_DIR/submodules/metric_modules/metric3d/mono"
if [ ! -e "$MONO_DIR/mono_utils" ]; then
    ln -s "$MONO_DIR/utils" "$MONO_DIR/mono_utils"
    echo "  Created mono_utils -> utils symlink"
else
    echo "  mono_utils already exists, skipping"
fi

echo "=== Verifying imports ==="
cd "$SCRIPT_DIR"
python -c "
import sys, os
sys.path.insert(0, 'submodules/')
import torch

results = {}
pkgs = {
    'diff_surfel_rasterization': 'diff_surfel_rasterization',
    'droid_backends': 'droid_backends',
    'mmengine': 'mmengine',
    'lpips': 'lpips',
    'kornia': 'kornia',
    'torch_scatter': 'torch_scatter',
    'open3d': 'open3d',
    'psutil': 'psutil',
}
for name, mod in pkgs.items():
    try:
        __import__(mod)
        print(f'  OK: {name}')
    except ImportError as e:
        print(f'  FAIL: {name} - {e}')

from metric_modules import Metric
print(f'  OK: metric_modules')

import onnxruntime as ort
providers = ort.get_available_providers()
gpu_ok = 'CUDAExecutionProvider' in providers
print(f'  onnxruntime GPU: {\"OK\" if gpu_ok else \"FAIL - no CUDAExecutionProvider\"}')
print(f'  providers: {providers}')
"

echo ""
echo "=== RTX 5090 setup complete! ==="
echo "=== Run with: conda activate vings_vio && python scripts/run.py ... ==="
