#!/bin/bash

#SBATCH --account=ligang
#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=g0168
#SBATCH --time=1:00:00
#SBATCH --job-name=ngp_plane_gpu1_btch1_iters30000

module load nvidia/cuda/11.6
module load scl/gcc10.2

cd $SLURM_SUBMIT_DIR
cd ../../
source activate negs

python train.py -s /project/ligang/NeRF/data/anisotropic/plane/ -m exps/trial_asyn_plane/alpha2 --eval --bound 2.0 --densify_ema --densify_decay 0.8 --offset 0.01 --tensor_fusion;
python metrics2.py -s /project/ligang/NeRF/data/anisotropic/plane/ -m exps/trial_asyn_plane/alpha2 --eval --bound 2.0 --offset 0.01 --tensor_fusion --iteration 40000;



