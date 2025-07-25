#!/bin/bash

#SBATCH --account=ligang
#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=g0150
#SBATCH --time=2:00:00
#SBATCH --job-name=ngp_nerf_gpu1_btch1_iters30000

module load nvidia/cuda/11.6
module load scl/gcc10.2

cd $SLURM_SUBMIT_DIR
cd ../../
source activate negs

python metrics2.py -s /project/ligang/NeRF/data/nerf_synthetic/ship/ -m exps/trial_syn_ship/final2 --eval -w --offset 0.01 --bound 1.5 --tensor_fusion --iteration 40000;
python metrics2.py -s /project/ligang/NeRF/data/nerf_synthetic/materials/ -m exps/trial_syn_materials/final2 --eval -w --offset 0.01 --bound 1.5 --tensor_fusion --iteration 40000;
python metrics2.py -s /project/ligang/NeRF/data/nerf_synthetic/lego/ -m exps/trial_syn_lego/final2 --eval -w --offset 0.01 --bound 1.5 --tensor_fusion --iteration 40000;
python metrics2.py -s /project/ligang/NeRF/data/nerf_synthetic/hotdog/ -m exps/trial_syn_hotdog/final2 --eval -w --bound 1.5 --tensor_fusion --iteration 40000;
python metrics2.py -s /project/ligang/NeRF/data/nerf_synthetic/drums/ -m exps/trial_syn_drums/final2 --eval -w --bound 1.5 --tensor_fusion --iteration 40000;
python metrics2.py -s /project/ligang/NeRF/data/nerf_synthetic/chair/ -m exps/trial_syn_chair/final2 --eval -w --bound 1.5 --tensor_fusion --iteration 40000;
python metrics2.py -s /project/ligang/NeRF/data/nerf_synthetic/mic/ -m exps/trial_syn_mic/final2 --eval -w --bound 1.5 --tensor_fusion --iteration 40000;
python metrics2.py -s /project/ligang/NeRF/data/nerf_synthetic/ficus/ -m exps/trial_syn_ficus/final2 --eval -w --offset 0.01 --bound 1.5 --tensor_fusion --iteration 40000;

