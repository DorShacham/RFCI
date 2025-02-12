#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import eigsh

from IQH_state import *
from flux_attch import *

Nx = 2
Ny = 6
path = str(f"./results/local_potential/interaction_shift/Nx-{Nx}_Ny-{Ny}")
os.makedirs(path, exist_ok=True)

pot_strength_list = np.linspace(0,0.01,num=25+1)

H_non_interacting = sparse.load_npz(str(f'data/matrix/H_Nx-{Nx}_Ny-{Ny}.npz'))
interaction = sparse.load_npz(str(f'data/matrix/interactions_Nx-{Nx}_Ny-{Ny}.npz'))
local_pot = sparse.load_npz(str(f'data/matrix/local_potential_Nx-{Nx}_Ny-{Ny}.npz'))

for interaction_strength in ([1e3]):
# for interaction_strength in ([1e-1, 1e0, 2e0, 1e1, 1e2, 1e3]):
    print(f"interaction strength = {interaction_strength}")
    eigenvalues_list = []
    H = H_non_interacting + interaction_strength * interaction
    for pot_strength in tqdm(pot_strength_list):
        
        H += pot_strength * local_pot
        eigenvalues = eigsh(H, k=7, which='SA', return_eigenvectors=False)
        eigenvalues_list.append(np.sort(eigenvalues))

    eigenvalues_list = np.array(eigenvalues_list) 
    plt.figure()
    plt.plot(pot_strength_list,eigenvalues_list[:,0], "-.")
    plt.plot(pot_strength_list,eigenvalues_list[:,1], "-.")
    plt.plot(pot_strength_list,eigenvalues_list[:,2], "-.")
    plt.plot(pot_strength_list,eigenvalues_list[:,3], "-.")
    plt.plot(pot_strength_list,eigenvalues_list[:,4], "-.")
    plt.plot(pot_strength_list,eigenvalues_list[:,5], "-.")
    plt.plot(pot_strength_list,eigenvalues_list[:,6], "-.")
    plt.grid()
    
    plt.title(f"interaction strentgh\n of {interaction_strength} for ({Nx,Ny}) lattice \n(first 7 eigenvalues)")
    plt.savefig(path + str(f"/interaction-{interaction_strength}_k=7.jpg"))


    plt.figure()
    plt.plot(pot_strength_list,eigenvalues_list[:,0], "-.")
    plt.plot(pot_strength_list,eigenvalues_list[:,1], "-.")
    plt.plot(pot_strength_list,eigenvalues_list[:,2], "-.")
    plt.grid()

    plt.title(f"Spectral flow with interaction strentgh\n of {interaction_strength} for ({Nx,Ny}) lattice \n(first 3 eigenvalues)")
    plt.savefig(path + str(f"/interaction-{interaction_strength}_k=3.jpg"))
