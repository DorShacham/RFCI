#!/bin/bash

source /Local/ph_lindner/anaconda3/bin/activate rfci-env

# Run the Python script
python ~/scripts/RFCI/RFCI/build_matrix.py --matrix_type H -Nx 2 -Ny 6 -cpu 6 \
--save_path /storage/ph_lindner/dorsh/RFCI/data/matrix/spectral_flow \
--name $1 --phi $2

conda deactivate

