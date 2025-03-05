#!/bin/bash

source /Local/ph_lindner/anaconda3/bin/activate rfci-env
# Run the Python script
python ~/scripts/RFCI/RFCI/sp_condor.py -Nx $1 -Ny $2 -cpu 1

conda deactivate



