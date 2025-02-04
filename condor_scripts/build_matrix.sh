#!/bin/bash

source /Local/ph_lindner/anaconda3/bin/activate rfci-env

# Run the Python script
python ~/scripts/RFCI/RFCI/build_matrix.py -Nx 3 -Ny 3

conda deactivate

