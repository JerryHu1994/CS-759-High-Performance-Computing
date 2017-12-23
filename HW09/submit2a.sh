#!/bin/sh
#SBATCH --partition=slurm_shortgpu
#SBATCH --time=0-00:05:00 # run time in days-hh:mm:ss
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --error=/srv/home/jhu76/759--JerryHu1994/HW09/problem2a.err
#SBATCH --output=/srv/home/jhu76/759--JerryHu1994/HW09/problem2a.out
#SBATCH --gres=gpu:1

module load cuda
for((i=4;i<13;i++))
do
./problem2A $((2**i))
done
