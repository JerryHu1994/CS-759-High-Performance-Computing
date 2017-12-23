#!/bin/sh
#SBATCH --partition=slurm_shortgpu
#SBATCH --time=0-00:05:00 # run time in days-hh:mm:ss
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8  # Use 8 OpenMP threads (you can use however many you like)
#SBATCH --error=/srv/home/jhu76/759--JerryHu1994/HW12/check.err
#SBATCH --output=/srv/home/jhu76/759--JerryHu1994/HW12/check.out
#SBATCH --gres=gpu:1
module load cuda
module load openmpi/2.1.1
perl check_homework.pl

