#!/bin/bash

source /Local/ph_lindner/anaconda3/bin/activate rfci-env

# Run the Python script
python ~/scripts/RFCI/RFCI/ansatz_expressive_test.py -Nx $1 -Ny $2 -cpu $3

conda deactivate

