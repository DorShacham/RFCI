#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import eigsh

from IQH_state import *
from flux_attch import *

Nx = 2
Ny = 6
step = 1
phi_list = np.array(range(0,72 + 1, step)) / 72 * 3

interaction = sparse.load_npz(str(f'data/matrix/interactions_Nx-{Nx}_Ny-{Ny}.npz'))


for interaction_strength in tqdm([1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1 ,9e-1 ,1,1,10,100,1000]):
    print(f"interaction strength = {interaction_strength}")
    eigenvalues_list = []
    for i in (range(0,72 + 1, step)):
        H_non_interacting = sparse.load_npz(str(f'data/matrix/spectral_flow/H_Nx-{Nx}_Ny-{Ny}_{i}.npz'))
        H = H_non_interacting + interaction_strength * interaction
        eigenvalues = eigsh(H, k=7, which='SA', return_eigenvectors=False)
        eigenvalues_list.append(np.sort(eigenvalues))
        print(i)

    eigenvalues_list = np.array(eigenvalues_list) 
    plt.figure()
    plt.plot(phi_list,eigenvalues_list[:,0], "-.")
    plt.plot(phi_list,eigenvalues_list[:,1], "-.")
    plt.plot(phi_list,eigenvalues_list[:,2], "-.")
    plt.plot(phi_list,eigenvalues_list[:,3], "-.")
    plt.plot(phi_list,eigenvalues_list[:,4], "-.")
    plt.plot(phi_list,eigenvalues_list[:,5], "-.")
    plt.plot(phi_list,eigenvalues_list[:,6], "-.")
    plt.grid()
    
    plt.title(f"Spectral flow with interaction strentgh\n of {interaction_strength} for ({Nx,Ny}) lattice \n(first 7 eigenvalues shifted by the lowest value)")
    plt.savefig(f"./results/spectral_flow/interaction_shift/Nx-{Nx}_Ny-{Ny}/interaction-{interaction_strength}_k=7.jpg")


    plt.figure()
    plt.plot(phi_list,eigenvalues_list[:,0], "-.")
    plt.plot(phi_list,eigenvalues_list[:,1], "-.")
    plt.plot(phi_list,eigenvalues_list[:,2], "-.")
    plt.grid()

    plt.title(f"Spectral flow with interaction strentgh\n of {interaction_strength} for ({Nx,Ny}) lattice \n(first 3 eigenvalues shifted by the lowest value)")
    plt.savefig(f"./results/spectral_flow/interaction_shift/Nx-{Nx}_Ny-{Ny}/interaction-{interaction_strength}_k=3.jpg")
