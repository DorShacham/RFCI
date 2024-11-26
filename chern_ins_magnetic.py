#%%

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from IQH_state import *
from flux_attch import *

def add_magnetic_field(H_real_space, p, q, Nx, Ny, cites_per_uc):
# adding vector potential A = (0,Bx,0) in Landuo gague
    # for x in range(Nx):
    #     for y in range(Ny):
    #         for sublattice in range(cites_per_uc):
    #             # cite_index = cites_per_uc * (Ny * x + y ) + sublattice
    #             cite_A_index = cites_per_uc * (Ny * x + y ) + sublattice
    #             cite_B_index = cites_per_uc * (Ny * x + (y + 1) % Ny) + sublattice
    #             hopping_phase = np.exp(1j * 2 * np.pi * (p / q) * x)
    #             H_real_space[cite_B_index,cite_A_index] *= hopping_phase
    #             H_real_space[cite_A_index,cite_B_index] *= hopping_phase.conjugate()
    # return H_real_space

# adding vector potential A = (By,0,0) 
    for x in range(Nx):
        for y in range(Ny):
            for sublattice in range(cites_per_uc):
            # cite_index = cites_per_uc * (Ny * x + y ) + sublattice
                cite_A_index = cites_per_uc * (Ny * x + y ) + sublattice
                cite_B_index = cites_per_uc * (Ny * ((x + 1) % Nx) + y) + sublattice
                hopping_phase = np.exp(1j * 2 * np.pi * (p / q) * y)
                H_real_space[cite_B_index,cite_A_index] *= hopping_phase
                H_real_space[cite_A_index,cite_B_index] *= hopping_phase.conjugate()
    return H_real_space

def magnetic_FT(H_real_space,Nx,Ny,q,cites_per_uc):
    N = Nx * Ny
    dft_matrix = np.kron(dft(Nx) ,(np.kron(dft(Ny // q),np.eye(cites_per_uc * q)))) / np.sqrt(N // q)
    H_k_space = np.matmul(np.matmul(dft_matrix,H_real_space), dft_matrix.T.conjugate())
    return H_k_space

def Hofstadter_butterfly(Nx,Ny, q = 100):
    H_real_space_vanila = build_basic_H(Nx,Ny)
    
    plt.figure()
    for p in tqdm(range(q)):
        H_real_space = add_magnetic_field(np.array(H_real_space_vanila), p, q, Nx, Ny, 1)
        eig_val, eig_vec = np.linalg.eigh(H_real_space)
        for eig in eig_val:
            plt.plot(p / q, eig, '.',color = 'black', markersize=1)  # 'bo' specifies blue circular markers

    plt.show()

def eigen_value_test(Nx,Ny,p,q, model = 'basic'):
    if model == 'basic':
        cites_per_uc = 1
        H_real_space = build_basic_H(Nx,Ny)
    elif model == 'chern':
        cites_per_uc = 2
        H_real_space = build_H(Nx,Ny)

    H_real_space = add_magnetic_field(np.array(H_real_space_vanila), p, q, Nx, Ny, cites_per_uc)
    # plt.matshow(np.abs(H_real_space))

    eig_val_original, eig_vec = np.linalg.eigh(H_real_space)

    H_k_space = magnetic_FT(H_real_space, Nx, Ny, q, cites_per_uc=1)
    # plt.matshow(np.abs(H_k_space))
    E = np.zeros((cites_per_uc * q, Nx, Ny // q))
    for kx in range(Nx):
        for ky in range(Ny // q):
            unit_cell = H_k_space[(cites_per_uc * q) * ((Ny // q) * kx + ky) : (cites_per_uc * q) * ((Ny // q) * kx + ky + 1),(cites_per_uc * q) * ((Ny // q) * kx + ky) : (cites_per_uc * q) * ((Ny // q) * kx + ky + 1)]
        #    print(f"---{np.shape(unit_cell)}----\n\n")
        #    print(unit_cell)

            eig_val, eig_vec = np.linalg.eigh(unit_cell)
            E[:,kx,ky] = eig_val

    eig_val_magnetic_FT = np.sort(np.reshape(E,(len(eig_val_original))))
    print(eig_val_original)
    print("\n\n -------------- \n\n")
    print(eig_val_magnetic_FT)
    print(np.sum(np.abs(eig_val_original - eig_val_magnetic_FT)))

def plot_BZ(Nx, Ny, p, q, model = 'basic'):
    if model == 'basic':
        cites_per_uc = 1
        H_real_space = build_basic_H(Nx,Ny)
    elif model == 'chern':
        cites_per_uc = 2
        H_real_space = build_H(Nx,Ny)
    
    H_real_space = add_magnetic_field(np.array(H_real_space),p,q,Nx,Ny,cites_per_uc)
    H_k_space = magnetic_FT(H_real_space, Nx, Ny, q, cites_per_uc)
    # plt.matshow(np.abs(H_k_space))

    Kx = np.linspace(0, 2 * np.pi,num=Nx + 1,endpoint=True)
    Ky = np.linspace(0, 2 * np.pi / q,num=Ny // q + 1,endpoint=True)
    
    E = np.zeros((cites_per_uc * q, Nx + 1, Ny // q + 1))
    for kx in range(Nx + 1):
        for ky in range(Ny // q + 1):
            kx_m = kx % Nx
            ky_m = ky % (Ny // q)
            unit_cell = H_k_space[(cites_per_uc * q) * ((Ny // q) * kx_m + ky_m) : (cites_per_uc * q) * ((Ny // q) * kx_m + ky_m + 1),(cites_per_uc * q) * ((Ny // q) * kx_m + ky_m) : (cites_per_uc * q) * ((Ny // q) * kx_m + ky_m + 1)]
    
            eig_val, eig_vec = np.linalg.eigh(unit_cell)
            E[:,kx,ky] = eig_val


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    kx, ky = np.meshgrid(Kx, Ky,indexing="ij")

    for r in range(cites_per_uc * q):
        color = plt.cm.tab10(r)  # Tab10 colormap
        ax.plot_surface(kx, ky, E[r,:,:], color=color)

    ax.view_init(elev=0, azim=90)  # Elevation = 30 degrees, Azimuth = 45 degrees
        # Labels and title
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('E(kx, ky)')
    ax.set_title('3D Plot of E(kx, ky)')

    # Show the plot
    plt.show()
    return E


def chern_number(Nx,Ny,p,q, model = 'basic'):
    if model == 'basic':
        cites_per_uc = 1
        H_real_space = build_basic_H(Nx,Ny)
    elif model == 'chern':
        cites_per_uc = 2
        H_real_space = build_H(Nx,Ny)

    H_real_space = add_magnetic_field(np.array(H_real_space), p, q, Nx, Ny, cites_per_uc)

    H_k_space = magnetic_FT(H_real_space, Nx, Ny, q, cites_per_uc)
    E = np.zeros((cites_per_uc * q, Nx, Ny // q))
    u_k = np.zeros((cites_per_uc * q, Nx, Ny // q, (cites_per_uc * q)), dtype=complex)
    for kx in range(Nx):
        for ky in range(Ny // q):
            unit_cell = H_k_space[(cites_per_uc * q) * ((Ny // q) * kx + ky) : (cites_per_uc * q) * ((Ny // q) * kx + ky + 1),(cites_per_uc * q) * ((Ny // q) * kx + ky) : (cites_per_uc * q) * ((Ny // q) * kx + ky + 1)]

            eig_val, eig_vec = np.linalg.eigh(unit_cell)
            E[:,kx,ky] = eig_val
            u_k[:,kx,ky,:] = eig_vec.T

    def U(n,mu,kx,ky):
        u = np.matmul(u_k[n,(kx % Nx),(ky % (Ny // q)),:].T.conjugate(), u_k[n,(kx + mu[0]) % Nx,(ky + mu[1]) % (Ny // q),:])
        u /= np.abs(u)
        return u
    
    C = np.zeros(q * cites_per_uc)
    for kx in range(Nx):
        for ky in range(Ny // q):
            for n in range(q * cites_per_uc):
                F12 = np.angle( U(n,[1,0],kx,ky) * U(n,[0,1],kx + 1,ky) / U(n,[1,0],kx,ky + 1) / U(n,[0,1],kx,ky)) / (2 * np.pi)
                C[n] += F12
    return C
#%%
# Hofstadter_butterfly(Nx = 10, Ny = 10, q = 100)
#%%
# eigen_value_test(Nx=24,Ny=24,p=1,q=3, model = 'chern')

#%%
Nx = 3
Ny = 3
p = 1
q = 3
# model = 'basic'
model = 'chern'

E = plot_BZ(Nx, Ny, p,q,model)
C = chern_number(Nx, Ny, p = p,q = q,model = model)
print(C)
for n in range(len(C)):
    print(f"C_{n} = {C[n]}")

# %%
# Let us simulate taking a full magnetic chern band state and then increading the field (or turnning it off)
# s.t it will be in 1/3 of the non magnetic chern band and add 2 flux per electron 
# and calculate the energy

Nx = 3
Ny = 3
p = 1
q = 3

# electron number fill one 'Landu' level
n = Nx * Ny // q

H_real_space_v = build_H(Nx,Ny)
H_real_space_magnetic = add_magnetic_field(np.array(H_real_space), p, q, Nx, Ny, cites_per_uc = 2)
state, mps = create_IQH_in_extendend_lattice(Nx,Ny,n,extention_factor = 1, band_energy = 1, H_sb = H_real_space_magnetic)
state = flux_attch_2_compact_state(state,mps,Ny)


# calculting the energy on the interaction with out mangetic field many body H
# <psi|H_many_body|psi> / <psi|psi>
E = np.matmul(state.T.conjugate(), mps.H_manby_body(H_real_space,state)) / np.linalg.norm(state)
print(E)