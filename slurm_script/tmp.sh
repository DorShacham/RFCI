#!/bin/bash

##Resource Request

#SBATCH --job-name spectral_flow
#SBATCH --mail-user=dor-hay.sha@campus.technion.ac.il
#SBATCH --mail-type=ALL           # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output /home/dor-hay.sha/master/RFCI/slurm_script/output/spectral_flow-result-%j.out   ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH -c 7
#SBATCH --time=0-4:10:00  ## time for analysis (day-hour:min:sec)

##Load the CUDA module
module load cuda

eval "$(conda shell.bash hook)"
conda activate rfci-env

## Run the script
nvidia-smi
python playground.py
echo Done