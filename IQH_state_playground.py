#%% 
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
from tqdm import tqdm
from IQH_state import *
from flux_attch import *
from exact_diagnolization import *
from qiskit_simulation import *
# %%

Nx = 3
Ny = 3
N = 2 * Nx * Ny
n = N // 6
# state, mps = create_IQH_in_extendend_lattice(Nx,Ny,n,extention_factor = 1, band_energy = 1)
# print_mp_state(state, Nx, Ny, mps)
# state_vector = state_2_full_state_vector(state, mps)
# print_state_vector(state_vector,Nx,Ny)


eigenvalues, eigenvectors = exact_diagnolization(Nx, Ny, n=n, band_energy=1, interaction_strength=1e-1, multi_process=False, save_result=False,k=4)