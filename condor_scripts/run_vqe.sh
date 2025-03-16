#!/bin/bash

source /Local/ph_lindner/anaconda3/bin/activate rfci-env

# Run the Python script
python ~/scripts/RFCI/RFCI/vqe_simulation.py --config_path configs/jax/Nx-$1_Ny-$2_layer-$3_flux-$4.yaml --log -cpu 1

conda deactivate

