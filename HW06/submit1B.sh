#!/bin/sh
#SBATCH --partition=slurm_shortgpu
#SBATCH --nodes=1
#SBATCH --tasks=40
#SBATCH --cpus-per-task=1
#SBATCH -e job_err
#SBATCH -o job_out
###SBATCH --gres=gpu:1 # not needed for OpenMP

cd $SLURM_SUBMIT_DIR
for((i=1; i<21;i++))
do
	./problem1B $i	
done
mv job_out problem1B.out
mv job_err problem1B.err
uname -n
