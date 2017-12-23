#!/bin/bash
#SBATCH -p slurm_shortgpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH -e job_err
#SBATCH -o job_out
#SBATCH --gres=gpu:1
cd $SLURM_SUBMIT_DIR
./problem3 1024 128
mv job_out problem3.out
mv job_err problem3.err
