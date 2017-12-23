#!/bin/sh
#SBATCH --partition=slurm_shortgpu
#SBATCH --time=0-00:05:00 # run time in days-hh:mm:ss
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8  # Use 8 OpenMP threads (you can use however many you like)
#SBATCH --error=/srv/home/jhu76/759--JerryHu1994/HW12/problem2.err
#SBATCH --output=/srv/home/jhu76/759--JerryHu1994/HW12/problem2.out
#SBATCH --gres=gpu:1
module load cuda
module load openmpi/2.1.1
mpiexec -np 2 ./problem2 8096
#for((i=11;i<25;i++))
#do for((j=0;j<10;j++))
#do
#mpiexec -np 2 ./problem2 $((2**i))
#done
#done
