#!/bin/bash

# Define the range for phi using np.linspace equivalent in Bash
START=0
STOP=3
NUM=73  # 72 intervals + 1 endpoint

# Calculate step size
STEP=$(echo "scale=10; ($STOP - $START)/($NUM - 1)" | bc)

# Loop through the range and execute the Python script for each phi value
for i in $(seq 0 $(($NUM - 1))); do
    PHI=$(echo "scale=10; $START + $i * $STEP" | bc)
    python build_matrix.py -Nx 2 -Ny 6 -cpu 10 --matrix_type H --phi "$PHI"
done
