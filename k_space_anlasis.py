#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import eigsh
import os
from exact_diagnolization import *
from IQH_state import *


def plot_k_space(Nx,Ny, interaction_strength, k, M = 0):
    if M !=0:
        H = build_non_interacting_H(Nx,Ny, H_sb=build_H(Nx,Ny,M=M))
    else:
        H = sparse.load_npz(str(f'data/matrix/H_Nx-{Nx}_Ny-{Ny}.npz'))
    interaction = sparse.load_npz(str(f'data/matrix/interactions_Nx-{Nx}_Ny-{Ny}.npz'))
    H = H + interaction_strength * interaction
    eigenvalues, eigenvectors = eigsh(H, k=k, which='SA')

    eigen_pairs = list(zip(eigenvalues, eigenvectors.T))
    eigen_pairs.sort(key=lambda x: x[0])
    eigenvalues, eigenvectors = zip(*eigen_pairs)
    eigenvalues = np.array(eigenvalues)
    eigenvectors = np.array(eigenvectors).T  # Transpose back to original shape

# plot spectrum
    path = str(f'results/k_space/Nx-{Nx}_Ny-{Ny}/')
    os.makedirs(path, exist_ok=True)
    plt.figure()
    plt.plot(np.ones(len(eigenvalues)), eigenvalues, ".")
    plt.title(f"interaction_strength = {interaction_strength}, M = {M}")
    plt.savefig(path + str(f'/spectrum_iter-{interaction_strength}_M-{M}.jpg'))

# plot k-space occupation
    N =  Nx * Ny
    n = N // 3
    mps = Multi_particle_state(2 * N, n)
    for i in range(k):
        state = eigenvectors[:,i]
        k_space_lower_band = project_on_band(state = state, mps = mps, band = -1, H = build_H(Nx,Ny), return_k_occupation=True)
        k_space_upper_band = project_on_band(state = state, mps = mps, band = 1, H = build_H(Nx,Ny), return_k_occupation=True)

        print(np.sum(k_space_lower_band))
        print(np.sum(k_space_upper_band))

        plt.figure()
        plt.plot(range(len(k_space_lower_band)), k_space_lower_band, "*", label = "lower band")
        plt.plot(range(len(k_space_lower_band)), k_space_upper_band, "*", label = "upper band")
        plt.grid()
        plt.legend()
        plt.xlabel(r"$k_y + N_y k_x$")
        plt.ylabel(r"$n(k_x,k_y)$")
        plt.title(f"interaction_strength = {interaction_strength}, M = {M}")
        plt.savefig(path + str(f'/n_k-{interaction_strength}_M-{M}_k-{i}.jpg'))

#%%
Nx = 3
Ny = 3
interaction_strength = 2
plot_k_space(Nx = Nx, Ny = Ny, interaction_strength=interaction_strength, k=4, M = 0)

#%%
Nx = 2
Ny = 6
interaction_strength_list = [2,0, 1e-3, 1e-1, 1e1, 1e2, 1e3]
for interaction_strength in interaction_strength_list:
    plot_k_space(Nx = Nx, Ny = Ny, interaction_strength=interaction_strength, k=4, M = 0)


#%%
Nx = 2
Ny = 6
interaction_strength = 2
M_list = [0.5, 1, 1.5]
for M in M_list:
    plot_k_space(Nx = Nx, Ny = Ny, interaction_strength=interaction_strength, k=4, M = M)

#%%
Nx = 3
Ny = 6
interaction_strength = 2
plot_k_space(Nx = Nx, Ny = Ny, interaction_strength=interaction_strength, k=4, M = 0)
