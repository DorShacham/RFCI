#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag, dft, expm
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import eigsh
from exact_diagnolization import *
from IQH_state import *

Nx = 2
Ny = 6

# H_sb = build_H(Nx,Ny, M = 10)
# H = build_non_interacting_H(Nx,Ny, H_sb=H_sb)
H = sparse.load_npz(str(f'data/matrix/H_Nx-{Nx}_Ny-{Ny}.npz'))
interaction = sparse.load_npz(str(f'data/matrix/interactions_Nx-{Nx}_Ny-{Ny}.npz'))

interaction_strength = 2

H = H  + interaction_strength * interaction
#%%
print("Diaganolizing")
eigenvalues, eigenvectors = eigsh(H, k=4, which='SA')
np.savez(f'data/states/Nx-{Nx}_Ny-{Ny}_k-4.npz', a=eigenvectors)
#%%
print("creating results")
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

#%%
print("Calculiting k-space")
M=0
# plot k-space occupation
N =  Nx * Ny
n = N // 3
mps = Multi_particle_state(2 * N, n)
for i in range(4):
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
