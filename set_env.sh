conda create --name vings_vio python=3.9.19
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.0.2 -f https://data.pyg.org/whl/torch-2.0.2+cu118.html
pip install -r requirements.txt

# Build dbaf.
cd submodules/dbaf
sudo python setup.py install

conda activate vings_vio
srun --time=4:00:00 -A ls_polle -n 1  --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_3090:1 --gres=gpumem:20g --pty bash
srun --time=4:00:00 -A ls_polle -n 1  --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --pty bash
module load stack/2024-04 cuda/11.8.0 

cd gtsam
mkdir build
cd build
cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.9.19
make python-install

module load stack/2024-04  gcc/12.2.0 openmpi/4.1.6 boost/1.83.0
module load cmake/3.30.5 
module load stack/2024-06  gcc/12.2.0  openmpi/4.1.6
module load gcc/8.5.0
python scripts/run.py configs/rtg/hotel.yaml
module load stack/2025-06 gcc/8.5.0 openmpi/4.1.6

python scripts/run.py configs/rtg/hotel.yaml



# USE THIS FOR RUN!!!!!!!!!!!!! (previous are some used for install, some need gcc8.5 some need 12.2)
# need 4090 for fp16 support
# srun --time=4:00:00 -A ls_polle -n 1  --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --pty bash
# srun --time=4:00:00 -A ls_polle -n 1  --cpus-per-task=4 --mem-per-cpu=10G --gpus=a100:1 --gres=gpumem:20g --pty bash
srun --time=4:00:00 -A ls_polle -n 1  --cpus-per-task=4 --mem-per-cpu=10G --pty bash
module load stack/2025-06 gcc/12.2.0   cuda/11.8.0 eth_proxy
conda activate vings_vio
python scripts/run_fastlivo.py --skip-slam --full-traj

python scripts/run_rpngar.py  --seqs table_01

python scripts/run_utmm.py  --full-traj --render-eval --skip-slam

python scripts/run.py configs/rpng/rpngar_table.yaml

zip -r VINGS-Mono.zip VINGS-Mono/

python scripts/run_rpngar.py  --full-traj --render-eval
--skip-slam
python scripts/run_rpngar.py  --seqs table_03 --full-traj --render-eval --skip-slam
python scripts/run_rpngar.py --seqs table_01  --skip-slam   --skip-slam
python scripts/run_rpngar.py --skip-slam 
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_rpngar.py --seqs table_01 --full-traj --render-eval"

sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_rpngar.py --seqs table_02 --full-traj --render-eval"

sbatch -A ls_polle -n 1 --cpus-per-task=8 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_rpngar.py --seqs table_03 --full-traj --render-eval"

sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_rpngar.py --seqs table_04 --full-traj --render-eval"

sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_rpngar.py --seqs table_05 --full-traj --render-eval"

sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_rpngar.py --seqs table_06 --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_rpngar.py --seqs table_07 --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_rpngar.py --seqs table_08 --full-traj --render-eval"


claude --dangerously-skip-permissions

previously the connection is cut off , so the previuos comamnd might not fully done. ideally still use the traj filter instead of the interpolation, since interpolation might not be accurate. the previuos command is to debug the full traj part, seems like the pose used is wrong, anyway, doublecheck and make sure the      
  interpolated pose is correct, use 'python scripts/run_fastlivo.py --seqs CBD_Building_01 ' command   with and without --full-traj to make sure after full the traj error is not so different (should be similar not 20cm v.s. 2m difference) , first try all you can do to make sure the interpolated pose is correct, if not, then try to debug the code.   
  then if it is good , try '
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_fastlivo.py --seqs CBD_Building_01 --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_fastlivo.py --seqs CBD_Building_02 --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_fastlivo.py --seqs Retail_Street --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_fastlivo.py --seqs HKU_Campus --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_fastlivo.py --seqs SYSU_01 --full-traj --render-eval"' and 'sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_rpngar.py --seqs table_01 --full-traj --render-eval" to make sure their full traj also good. 

python scripts/run_fastlivo.py --skip-slam --full-traj --render-eval


# sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_fastlivo.py --seqs HKU_Campus"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_fastlivo.py --seqs CBD_Building_01 --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_fastlivo.py --seqs CBD_Building_02 --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_fastlivo.py --seqs Retail_Street --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_fastlivo.py --seqs HKU_Campus --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_fastlivo.py --seqs SYSU_01 --full-traj --render-eval"


sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_utmm.py --seqs ego-centric-1 --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_utmm.py --seqs ego-centric-2 --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_utmm.py --seqs ego-drive --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_utmm.py --seqs fast-straight --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_utmm.py --seqs slow-straight-1 --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_utmm.py --seqs slow-straight-2 --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_utmm.py --seqs square-1 --full-traj --render-eval"
sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_utmm.py --seqs square-2 --full-traj --render-eval"

-render-eval --render-n 50

fast-straight
python scripts/run_euroc.py --seqs MH_01_easy V1_01_easy

EUROC_SEQUENCES=(
    MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult
    V1_01_easy V1_02_medium V1_03_difficult
    V2_01_easy V2_02_medium V2_03_difficult
)

for seq in "${EUROC_SEQUENCES[@]}"; do
    echo "Submitting job for $seq"

    sbatch -A ls_polle \
        -n 1 \
        --cpus-per-task=4 \
        --mem-per-cpu=10G \
        --gpus=rtx_4090:1 \
        --gres=gpumem:20g \
        --time=4:00:00 \
        --job-name="euroc_${seq}" \
        --output="logs/%x_%j.out" \
        --wrap="python scripts/run_euroc.py --seqs ${seq}"
done

sbatch -A ls_polle -n 1 --cpus-per-task=4 --mem-per-cpu=10G --gpus=rtx_4090:1 --gres=gpumem:20g --time=4:00:00 --wrap="python scripts/run_euroc.py --seqs ${EUROC_SEQUENCES}"


  # Skip SLAM (just set up output dirs/configs)
  python scripts/run_fastlivo.py --skip-slam

  # Custom output dir
  python run_fastlivo.py --output /path/to/output

  UTMM dataset:
  cd /cluster/project/cvg/zihzhu/VINGS-Mono/scripts

  # Run all sequences
  python run_utmm.py

  # Run specific sequences
  python scripts/run_utmm.py --seqs ego-centric-1 square-1

  # Skip SLAM
  python run_utmm.py --skip-slam

  Or run a single sequence directly:
  # Edit the root/save_dir in the config first, then:
  python run.py ../configs/fastlivo/fastlivo.yaml
  python run.py ../configs/utmm/utmm.yaml

  On Euler cluster (SLURM):
  # Example bsub/sbatch wrapper
  bsub -n 1 -W 24:00 -R "rusage[mem=32000,ngpus_excl_p=1]" \
      "cd /cluster/project/cvg/zihzhu/VINGS-Mono/scripts && python run_fastlivo.py --seqs HKU_Campus"