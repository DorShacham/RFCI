#!/bin/bash

source /Local/ph_lindner/anaconda3/bin/activate rfci-env

# Run the Python script
python ~/scripts/RFCI/RFCI/build_matrix.py -Nx 2 -Ny 6 -cpu 10
python ~/scripts/RFCI/RFCI/shared_tmp.py

conda deactivate

