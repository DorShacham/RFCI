#%%
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

interaction_strength = 0.1

H_new = H + interaction_strength * interaction
#%%
eigenvalues, eigenvectors = eigsh(H_new, k=4, which='SA')
#%%

np.savez(f'data/states/Nx-{Nx}_Ny-{Ny}_k-4.npz', a=eigenvectors)

#%%
from IQH_state import *
save_result = True
show_result = True
path = str(f'results/Exact_Diagnolization/Nx-{Nx}_Ny-{Ny}')
N = Nx * Ny
n = N // 3
band_energy = 1
interaction_strength = interaction_strength
mps = Multi_particle_state(2 * N, n)

eigen_pairs = list(zip(eigenvalues, eigenvectors.T))
eigen_pairs.sort(key=lambda x: x[0])
eigenvalues, eigenvectors = zip(*eigen_pairs)
eigenvalues = np.array(eigenvalues)
eigenvectors = np.array(eigenvectors).T  # Transpose back to original shape

if show_result or save_result:
    plt.figure()
    plt.plot(np.ones(len(eigenvalues)), eigenvalues, ".")
if save_result:
    plt.savefig(path + str('/eigenvalues.jpg'))
    print_mp_state(eigenvectors[:,0],Nx,Ny,mps,saveto= path + str("/ev0.jpg"))
    print_mp_state(eigenvectors[:,1],Nx,Ny,mps,saveto= path + str("/ev1.jpg"))
    print_mp_state(eigenvectors[:,2],Nx,Ny,mps,saveto= path + str("/ev2.jpg"))
    print_mp_state(eigenvectors[:,3],Nx,Ny,mps,saveto= path + str("/ev3.jpg"))
    
    with open(path + str('/data.txt'), 'w') as file:
        file_dict = {"Nx":Nx, "Ny":Ny, "n":n, "band_energy": band_energy, "interaction_strength":interaction_strength,"eigenvalues":eigenvalues}
        file.write(str(file_dict))
elif show_result:
    print_mp_state(eigenvectors[:,0],Nx,Ny,mps,saveto= None)
    print_mp_state(eigenvectors[:,1],Nx,Ny,mps,saveto= None)
    print_mp_state(eigenvectors[:,2],Nx,Ny,mps,saveto= None)
    print_mp_state(eigenvectors[:,3],Nx,Ny,mps,saveto= None)

