#!/bin/bash

#SBATCH --account=ligang
#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=g0177,g0150
#SBATCH --time=2:00:00
#SBATCH --job-name=ngp_teapot_gpu1_btch1_iters30000

module load nvidia/cuda/11.6
module load scl/gcc10.2

cd $SLURM_SUBMIT_DIR
cd ../../
source activate negs

python render.py -s /project/ligang/NeRF/data/anisotropic/teapot/ -m exps/trial_asyn_teapot/test1 --eval --bound 1.5 --offset 0.01 --tensor_fusion --iteration 40000 -w;
python render.py -s /project/ligang/NeRF/data/anisotropic/dishes/ -m exps/trial_asyn_dishes/test1 --eval --bound 1.5 --offset 0.01 --tensor_fusion --iteration 40000 -w;
python render.py -s /project/ligang/NeRF/data/anisotropic/ashtray/ -m exps/trial_asyn_ashtray/test1 --eval --bound 1.5 --offset 0.01 --tensor_fusion --iteration 40000 -w;
python render.py -s /project/ligang/NeRF/data/anisotropic/headphone/ -m exps/trial_asyn_headphone/test1 --eval --bound 3.5 --offset 0.01 --tensor_fusion --iteration 40000 -w;
python render.py -s /project/ligang/NeRF/data/anisotropic/jupyter/ -m exps/trial_asyn_jupyter/test1 --eval --bound 1.5  --offset 0.01 --tensor_fusion --iteration 40000 -w;
python render.py -s /project/ligang/NeRF/data/anisotropic/lock/ -m exps/trial_asyn_lock/test1 --eval --bound 2.5  --offset 0.01 --tensor_fusion --iteration 40000 -w;
python render.py -s /project/ligang/NeRF/data/anisotropic/plane/ -m exps/trial_asyn_plane/test1 --eval --bound 2  --offset 0.01 --tensor_fusion --iteration 40000 -w;
python render.py -s /project/ligang/NeRF/data/anisotropic/record/ -m exps/trial_asyn_record/test1 --eval --bound 1.5 --offset 0.01 --tensor_fusion --iteration 40000 -w;


