#!/bin/bash

#SBATCH --account=ligang
#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=g0177
#SBATCH --time=5:00:00
#SBATCH --job-name=ngp_playroom_gpu1_btch1_iters30000

module load nvidia/cuda/11.6
module load scl/gcc10.2

cd $SLURM_SUBMIT_DIR
cd ../../
source activate negs

python train.py -s /project/ligang/NeRF/data/tandt_db/db/playroom/ -m exps/trial_db_playroom/test_09 --eval --densify_ema --densify_decay 0.9 --contract;
python train.py -s /project/ligang/NeRF/data/tandt_db/db/playroom/ -m exps/trial_db_playroom/test_07 --eval --densify_ema --densify_decay 0.7 --contract;
python train.py -s /project/ligang/NeRF/data/tandt_db/db/playroom/ -m exps/trial_db_playroom/test_05 --eval --densify_ema --densify_decay 0.5 --contract;
python train.py -s /project/ligang/NeRF/data/tandt_db/db/playroom/ -m exps/trial_db_playroom/test_02 --eval --densify_ema --densify_decay 0.2 --contract;