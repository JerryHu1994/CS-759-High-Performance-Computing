#!/bin/bash
#SBATCH -p slurm_shortgpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH -e job_err
#SBATCH -o job_out
#SBATCH --gres=gpu:1
cd $SLURM_SUBMIT_DIR
for((i=10; i<21;i++))
do
./problem3 $((2**i)) 1024
done
mv job_out problem3_1024.out
mv job_err problem3_1024.err
