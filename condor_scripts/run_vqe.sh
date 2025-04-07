#!/bin/bash

source /Local/ph_lindner/anaconda3/bin/activate rfci-env

# Log compilation steps
export JAX_LOG_COMPILES=1

# Prevents JAX from preallocating 90% of RAM on CPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Use system malloc instead of JAX's default
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Dump compilation IRs to disk (advanced debugging)
export XLA_FLAGS=--xla_dump_to=/tmp/xla_dump

ulimit -c unlimited
# Run the Python script
taskset -c $5  -X faulthandler python ~/scripts/RFCI/RFCI/vqe_simulation.py --config_path configs/jax/Nx-$1_Ny-$2_layer-$3_flux-$4.yaml --log -cpu 1

conda deactivate

