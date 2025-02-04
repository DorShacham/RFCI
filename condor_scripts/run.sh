#!/bin/bash

source /Local/ph_lindner/anaconda3/bin/activate rfci-env

# Run the Python script
python ~/scripts/RFCI/RFCI/vqe_simulation.py --help

conda deactivate

