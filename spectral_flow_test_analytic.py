#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
import scipy
from tqdm import tqdm



Nx = 5
Ny = 3
n = Nx * Ny // 3
# parametrs of the model
N = Nx * Ny
phi = np.pi/4
t1 = 1
t2 = (2-np.sqrt(2))/2 * t1
M = 0
step = 1

def Ek(kx, ky):
    h11 = 2 * t2 * (np.cos(kx) - np.cos(ky)) + M
    h12 = t1 * np.exp(1j * phi) * (1 + np.exp(1j * (ky - kx))) + t1 * np.exp(-1j * phi) * (np.exp(1j * (ky)) + np.exp(1j * (- kx)))
    # h2 = np.array([[h11, h12], [np.conjugate(h12), -h11]])
    h2 = np.array([[h11, np.conjugate(h12)], [h12, -h11]])
    eig_val = np.linalg.eigvalsh(h2)
    assert(np.abs(abs(eig_val[0]) - abs(eig_val[1])) < 1e-8)
    return -abs(eig_val[0])

def many_body_E(phi_x = 0, phi_y = 0, k = 7):
    Kx = np.linspace(0, 2 * np.pi,num=Nx,endpoint=False)
    Ky = np.linspace(0, 2 * np.pi,num=Ny,endpoint=False)
    E = np.zeros((Nx,Ny))

    i = 0
    for x,kx in enumerate(Kx):
        for y,ky in enumerate(Ky):
            E[x,y] = Ek(kx + phi_x / Nx, ky + phi_y / Ny) + i * 1e-8

    E = np.reshape(E,(Nx * Ny))
    E = np.sort(E)
    
    objects = np.arange(Nx * Ny)
    perms = list((combinations(objects, n)))
    
    many_body_E_list = []
    for perm in perms:
        many_body_E_list.append(np.sum(E[np.array(perm)]))
    sorted(many_body_E_list)
    return many_body_E_list[:k]

phi_list = np.array(range(0,72 + 1, step)) / 72 * 3 * (2 * np.pi)

eigenvalues_list = []
for phi_x in phi_list:
    eigenvalues_list.append(many_body_E(phi_x = phi_x, phi_y = 0, k = 7))

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
plt.xlabel("phi_x")
plt.title(f"Spectral flow without interaction for ({Nx,Ny}) lattice \n(first 7 eigenvalues shifted by the lowest value)")
# plt.savefig(f"./results/spectral_flow/interaction_shift/Nx-{Nx}_Ny-{Ny}/interaction-{interaction_strength}_k=7.jpg")


plt.figure()
plt.plot(phi_list,eigenvalues_list[:,0], "-.")
plt.plot(phi_list,eigenvalues_list[:,1], "-.")
plt.plot(phi_list,eigenvalues_list[:,2], "-.")
plt.grid()
plt.xlabel("phi_x")
plt.title(f"Spectral flow without interaction for ({Nx,Ny}) lattice \n(first 3 eigenvalues shifted by the lowest value)")
# plt.savefig(f"./results/spectral_flow/interaction_shift/Nx-{Nx}_Ny-{Ny}/interaction-{interaction_strength}_k=3.jpg")


eigenvalues_list = []
for phi_y in phi_list:
    eigenvalues_list.append(many_body_E(phi_x = 0, phi_y = phi_y, k = 7))

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
plt.xlabel("phi_y")
plt.title(f"Spectral flow without interaction for ({Nx,Ny}) lattice \n(first 7 eigenvalues shifted by the lowest value)")
# plt.savefig(f"./results/spectral_flow/interaction_shift/Nx-{Nx}_Ny-{Ny}/interaction-{interaction_strength}_k=7.jpg")


plt.figure()
plt.plot(phi_list,eigenvalues_list[:,0], "-.")
plt.plot(phi_list,eigenvalues_list[:,1], "-.")
plt.plot(phi_list,eigenvalues_list[:,2], "-.")
plt.grid()
plt.xlabel("phi_y")
plt.title(f"Spectral flow without interaction for ({Nx,Ny}) lattice \n(first 3 eigenvalues shifted by the lowest value)")
# plt.savefig(f"./results/spectral_flow/interaction_shift/Nx-{Nx}_Ny-{Ny}/interaction-{interaction_strength}_k=3.jpg")
