#!/bin/sh
#SBATCH --partition=slurm_shortgpu
#SBATCH --nodes=1
#SBATCH --tasks=40
#SBATCH --cpus-per-task=1
#SBATCH -o job_out
#SBATCH -e job_err
###SBATCH --gres=gpu:1 # not needed for OpenMP
cd $SLURM_SUBMIT_DIR
for((i=1; i<41;i++))
do
	./problem2 $i	
done
mv job_out problem2a.out
mv job_err problem2a.err
