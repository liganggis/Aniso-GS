#!/bin/bash

#SBATCH --account=ligang
#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=g0168
#SBATCH --time=1:00:00
#SBATCH --job-name=ngp_dishes_gpu1_btch1_iters30000

module load nvidia/cuda/11.6
module load scl/gcc10.2

cd $SLURM_SUBMIT_DIR
cd ../../
source activate negs

python train.py -s /project/ligang/NeRF/data/anisotropic/dishes/ -m exps/trial_asyn_dishes/alpha --eval --bound 1.5 --densify_ema --densify_decay 0.8  --offset 0.01 --tensor_fusion;
python metrics2.py -s /project/ligang/NeRF/data/anisotropic/dishes/ -m exps/trial_asyn_dishes/alpha --eval --bound 1.5  --offset 0.01 --tensor_fusion --iteration 40000;

