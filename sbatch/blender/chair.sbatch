#!/bin/bash

#SBATCH --account=ligang
#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=g0150,g0158
#SBATCH --time=1:00:00
#SBATCH --job-name=ngp_chair_gpu1_btch1_iters30000

module load nvidia/cuda/11.6
module load scl/gcc10.2

cd $SLURM_SUBMIT_DIR
cd ../../
source activate negs

python train.py -s ../../data/nerf_synthetic/chair/ -m exps/nerf_synthetic/chair_full --eval -w --densify_ema --densify_decay 0.8 --bound 1.5
python metrics2.py -s ../../data/nerf_synthetic/chair/ -m exps/nerf_synthetic/chair_full --eval -w --bound 1.5 --iteration 40000
python render.py -s ../../data/nerf_synthetic/chair/ -m exps/nerf_synthetic/chair_full --eval -w --bound 1.5 --iteration 40000
