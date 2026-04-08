#!/bin/bash

#SBATCH --account=ligang
#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=g0173,g0158
#SBATCH --time=1:00:00
#SBATCH --job-name=ngp_ficus_gpu1_btch1_iters30000

module load nvidia/cuda/11.6
module load scl/gcc10.2

cd $SLURM_SUBMIT_DIR
cd ../../
source activate negs

python train.py -s /project/ligang/NeRF/data/nerf_synthetic/ficus/ -m exps/trial_syn_ficus/test1 --eval -w --offset 0.01 --densify_ema --densify_decay 0.8 --bound 1.5



python train.py -s ../../data/shiny_blender/car/ -m exps/shiny_blender/car/v1 --eval -w --offset 0.3 --densify_ema --densify_decay 0.8 --bound 1.5
python metrics2.py -s ../../data/shiny_blender/car/ -m exps/shiny_blender/car/v1 --eval --offset 0.3 --bound 1.5 --iteration 40000
python render.py -s ../../data/shiny_blender/car/ -m exps/shiny_blender/car/v1 --eval --offset 0.3 --bound 1.5 --iteration 40000


python train.py -s ../../data/shiny_blender/ball/ -m exps/shiny_blender/ball/v1 --eval -w --offset 0.3 --densify_ema --densify_decay 0.8 --bound 1.5
python metrics2.py -s ../../data/shiny_blender/ball/ -m exps/shiny_blender/ball/v1 --eval --offset 0.3 --bound 1.5 --iteration 40000
python render.py -s ../../data/shiny_blender/ball/ -m exps/shiny_blender/ball/v1 --eval --offset 0.3 --bound 1.5


python train.py -s ../../data/shiny_blender/coffee/ -m exps/shiny_blender/coffee/v1 --eval -w --offset 0.3 --densify_ema --densify_decay 0.8 --bound 1.6
python metrics2.py -s ../../data/shiny_blender/coffee/ -m exps/shiny_blender/coffee/v1 --eval --offset 0.3 --bound 1.6 --iteration 40000


python train.py -s ../../data/shiny_blender/helmet/ -m exps/shiny_blender/helmet/v1 --eval -w --offset 0.3 --densify_ema --densify_decay 0.8 --bound 1.4
python metrics2.py -s ../../data/shiny_blender/helmet/ -m exps/shiny_blender/helmet/v1 --eval --offset 0.3 --bound 1.4 --iteration 40000


python train.py -s ../../data/shiny_blender/teapot/ -m exps/shiny_blender/teapot/v1 --eval -w --offset 0.3 --densify_ema --densify_decay 0.8 --bound 2.0
python metrics2.py -s ../../data/shiny_blender/teapot/ -m exps/shiny_blender/teapot/v1 --eval --offset 0.3 --bound 2.0 --iteration 40000


python train.py -s ../../data/shiny_blender/toaster/ -m exps/shiny_blender/toaster/v1 --eval -w --offset 0.3 --densify_ema --densify_decay 0.8 --bound 1.5
python metrics2.py -s ../../data/shiny_blender/toaster/ -m exps/shiny_blender/toaster/v1 --eval --offset 0.3 --bound 1.5 --iteration 40000