#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import eigsh
import os
from exact_diagnolization import *
from IQH_state import *

#%%
if __name__ == "__main__":
    Nx = 3
    Ny = 3
    interaction_strength = 2
    t1 = 1
    t2 = (2-np.sqrt(2))/2
    M_over_4t2 = np.linspace(0,1.5,20)
    interaction = sparse.load_npz(str(f'data/matrix/interactions_Nx-{Nx}_Ny-{Ny}.npz'))

            
    H = sparse.load_npz(str(f'data/matrix/H_Nx-{Nx}_Ny-{Ny}.npz'))
    H = H + interaction_strength * interaction
    eigenvalues, eigenvectors = eigsh(H, k=4, which='SA')

    gap_list = []
    for M in tqdm(M_over_4t2):
        H = build_non_interacting_H(Nx,Ny, H_sb=build_H(Nx,Ny,M= 4 * t2 * M), multi_process=False)
        H = H + interaction_strength * interaction
        eigenvalues = eigsh(H, k=4, which='SA', return_eigenvectors=False)
        eigenvalues = np.sort(np.array(eigenvalues))
        gap_list.append(eigenvalues[3] - eigenvalues[2])

    plt.figure()
    plt.plot(M_over_4t2, gap_list, ".")
    plt.grid()
    plt.xlabel(r"$\frac{M}{4t_2}$")
    plt.ylabel(r"$\frac{\Delta}{t_1}$")
    plt.title(f"gap as a function of M for ({Nx},{Ny})")
    plt.savefig('results/gap_M.jpg')
