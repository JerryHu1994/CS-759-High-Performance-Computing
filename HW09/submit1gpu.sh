#!/bin/sh
#SBATCH --partition=slurm_shortgpu
#SBATCH --time=0-00:05:00 # run time in days-hh:mm:ss
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --error=/srv/home/jhu76/759--JerryHu1994/HW09/problem1gpu.err
#SBATCH --output=/srv/home/jhu76/759--JerryHu1994/HW09/problem1gpu.out
#SBATCH --gres=gpu:1

module load cuda
for((i=3;i<9;i++))
do
./problem1gpu $((10**i))
done
