#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=work_dirs/srun_output/ice_lake_mask2former_swinb12_it15k_%j.out
#SBATCH --error=work_dirs/srun_output/ice_lake_mask2former_swinb12_it15k_%j.err
#SBATCH --time=02:00:00
#SBATCH --job-name=mask2former_seg
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=booster

# define available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

sh tools/dist_train.sh configs/mask2former/mask2former_swin-b-in22k-384x384-pre_4xb2-15k_ice_lake-1024x1024.py 4 --work-dir work_dirs/mask2former_swin-b-in22k-384x384-pre_4xb2-15k_ice_lake-1024x1024 #--resume
