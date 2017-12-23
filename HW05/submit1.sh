#!/bin/sh
#SBATCH --partition=slurm_shortgpu
#SBATCH --nodes=1
#SBATCH --tasks=40
#SBATCH --cpus-per-task=1
#SBATCH -e job_err
#SBATCH -o job_out
###SBATCH --gres=gpu:1 # not needed for OpenMP

cd $SLURM_SUBMIT_DIR
threads=(1, 2, 4, 6, 8, 10, 14, 16, 20)
for i in "${threads[@]}"
do
	./problem1 $i	
done
mv job_out problem1.out
mv job_err problem1.err
