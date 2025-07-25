#!/bin/bash

#SBATCH --account=ligang
#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=g0168
#SBATCH --time=1:00:00
#SBATCH --job-name=ngp_bonsai_gpu1_btch1_iters30000

module load nvidia/cuda/11.6
module load scl/gcc10.2

cd $SLURM_SUBMIT_DIR
cd ../../
source activate negs

python train.py -s /project/ligang/NeRF/data/360_v2/bonsai/ -m exps/trial_360_bonsai/no_opacity  -r 2 --eval --contract --densify_ema --densify_decay 0.8 --tensor_fusion;
python metrics2.py -s /project/ligang/NeRF/data/360_v2/bonsai/ -m exps/trial_360_bonsai/no_opacity -r 2 --eval --contract --iteration 40000;