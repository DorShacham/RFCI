#!/bin/bash

source /Local/ph_lindner/anaconda3/bin/activate rfci-env

# Run the Python script
python -u  ~/scripts/RFCI/RFCI/vqe_simulation.py --log  --config_path ~/scripts/RFCI/RFCI/configs/jax/config_template.yaml 

conda deactivate

