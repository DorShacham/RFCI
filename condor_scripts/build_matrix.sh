#!/bin/bash

TARGET_FILE="/storage/ph_lindner/dorsh/RFCI/data/matrix/spectral_flow/H_Nx-2_Ny-6_$1.npz"

# Check if the file exists
if [ -f "$TARGET_FILE" ]; then
    echo "$TARGET_FILE exists."
else
    echo "$TARGET_FILE does not exist. Running script.py..."
    
    source /Local/ph_lindner/anaconda3/bin/activate rfci-env
    # Run the Python script
    python ~/scripts/RFCI/RFCI/build_matrix.py --matrix_type H -Nx 3 -Ny 6 -cpu 6 \
    --save_path /storage/ph_lindner/dorsh/RFCI/data/matrix/spectral_flow \
    --name $1 --phi $2

    conda deactivate
fi



