#!/bin/bash
#
#SBATCH --job-name=bash
#SBATCH --output=logs/my_logs.log
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=22:00:00
#SBATCH --mem-per-cpu=64000
export OMP_NUM_THREADS=28
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
HOME_PATH=/home/user/project/
cd $HOME_PATH
module load python3
source $HOME_PATH/venv/bin/activate
python script.py
