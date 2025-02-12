#%% 
import numpy as np
import scipy
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from IQH_state import *
from flux_attch import *
from exact_diagnolization import *
from qiskit_simulation import *
# %%

Nx = 3
Ny = 6
N = 2 * Nx * Ny
n = N // 6
# n = 4
mps = Multi_particle_state(N, n)

# state, mps = create_IQH_in_extendend_lattice(Nx,Ny,n,extention_factor = 1, band_energy = 1)
# print_mp_state(state, Nx, Ny, mps)
# state_vector = state_2_full_state_vector(state, mps)
# print_state_vector(state_vector,Nx,Ny)
loaded = np.load(f'data/states/Nx-{Nx}_Ny-{Ny}_k-4.npz')
eigenvectors = loaded['a']   
# H = sparse.load_npz(str(f'data/matrix/H_Nx-{Nx}_Ny-{Ny}.npz'))
# interaction = sparse.load_npz(str(f'data/matrix/interactions_Nx-{Nx}_Ny-{Ny}.npz'))

# interaction_strength = 1e3

# H = H + interaction_strength * interaction
# H = build_non_interacting_H(Nx, Ny, n)
# eigenvalues, eigenvectors = eigsh(H, k=4, which='SA')
# exact_diagnolization(Nx = Nx, Ny = Ny, interaction_strength=0, save_result=False,show_result=False)

state = eigenvectors[:,2]

k_space_lower_band = project_on_band(state = state, mps = mps, band = -1, H = build_H(Nx,Ny), return_k_occupation=True)
k_space_upper_band = project_on_band(state = state, mps = mps, band = 1, H = build_H(Nx,Ny), return_k_occupation=True)

print(np.sum(k_space_lower_band))
print(np.sum(k_space_upper_band))

plt.figure()
plt.plot(range(len(k_space_lower_band)), k_space_lower_band, "*", label = "lower band")
plt.plot(range(len(k_space_lower_band)), k_space_upper_band, "*", label = "upper band")
plt.grid()
plt.legend()