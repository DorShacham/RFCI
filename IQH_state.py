#%% 
import numpy as np
import matplotlib.pyplot as plt


#%% single electron Hamiltonian

# parametrs of the model
Nx = 3
Ny = 6
M = 0
phi = np.pi/4
t1 = 1
t2 = (2-np.sqrt(2))/2

# Building the single particle hamiltonian (h2)
# need to check if the gauge transformation is needed to adress
Kx = np.linspace(-np.pi,np.pi,num=Nx,endpoint=True)
Ky = np.linspace(-np.pi,np.pi,num=Ny,endpoint=True)

# Kx, Ky = np.meshgrid(kx,ky)
def build_h2(kx, ky):
    h11 = 2 * t2 * (np.cos(kx) - np.cos(ky)) + M
    h12 = t1 * np.exp(1j * phi) * (1 + np.exp(1j * (ky - kx))) + t1 * t1 * np.exp(-1j * phi) * (np.exp(1j * (ky)) + np.exp(1j * (- kx)))
    h2 = np.matrix([[h11, h12], [np.conjugate(h12), -h11]])
    return h2

# eigen states (projected on the lower energies) tensor (state index, A/B lattice, real space position x, real space position y)
eigen_states = np.zeros((Nx * Ny,2, Nx, Ny)) * 1j

state_index = 0
for kx in Kx:
    for ky in Ky:
        H_single_particle = build_h2(kx,ky)
        eig_val, eig_vec = np.linalg.eigh(H_single_particle)

        k_space = np.zeros((Nx,Ny))
        k_space[kx == Kx, ky == Ky] = 1
        real_space = np.fft.fft2(k_space)

        eigen_states[state_index, 0, :,:] = eig_vec[0,0] * real_space
        eigen_states[state_index, 1, :,:] = eig_vec[1,0] * real_space

        state_index += 1
    
