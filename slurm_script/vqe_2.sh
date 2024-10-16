#!/bin/bash

##Resource Request

#SBATCH --job-name vqe-2
#SBATCH --mail-user=dor-hay.sha@campus.technion.ac.il
#SBATCH --mail-type=ALL           # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output /home/dor-hay.sha/master/RFCI/slurm_script/output/vqe-2-result-%j.out   ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH -c 20
#SBATCH --time=0-10:10:00  ## time for analysis (day-hour:min:sec)

##Load the CUDA module
module load cuda

eval "$(conda shell.bash hook)"
conda activate rfci-env

## Run the script
nvidia-smi
python vqe_simulation.py --config_path ./configs/config_2.yaml
echo Done