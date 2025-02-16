#%%
import os
import platform
import multiprocessing


# Check the operating system
if platform.system() == "Linux":
    # Set environment variables to limit CPU usage on Linux
    os.environ["OMP_NUM_THREADS"] = "10"
    os.environ["MKL_NUM_THREADS"] = "10"
    os.environ["NUMEXPR_NUM_THREADS"] = "10"
    os.environ["OPENBLAS_NUM_THREADS"] = "10"
    os.environ["JAX_NUM_THREADS"] = "10"
    print("CPU usage limited to 10 threads on Linux.")
elif platform.system() == "Darwin":
    # macOS-specific behavior (no limitation)
    print("Running on macOS. No CPU limitation applied.")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())
else:
    print("Operating system not recognized. No changes applied.")


import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag, dft, expm
from tqdm import tqdm
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import concurrent.futures
from functools import partial
from multiprocessing import Manager, shared_memory
import os
import gc
from collections import ChainMap

Nx = 3
Ny = 6

H = sparse.load_npz(str(f'data/matrix/H_Nx-{Nx}_Ny-{Ny}.npz'))
interaction = sparse.load_npz(str(f'data/matrix/interactions_Nx-{Nx}_Ny-{Ny}.npz'))

interaction_strength = 2

H_new = H + interaction_strength * interaction

loaded = np.load(f'data/states/Nx-{Nx}_Ny-{Ny}_k-4.npz')
eigenvectors = loaded['a']   
#%%
# eigenvalues, eigenvectors = eigsh(H_new, k=4, which='SA')
# np.savez(f'data/states/Nx-{Nx}_Ny-{Ny}_k-4.npz', a=eigenvectors)

for i in range(4):
    ev = eigenvectors[:,i].T.conjugate() @ (H_new @ eigenvectors[:,i])
    print(f'state {i} - norm={np.linalg.norm(eigenvectors[:,i]; <H> = {ev})}')

