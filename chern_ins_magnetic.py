#%%

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from IQH_state import *
from flux_attch import *

def build_basic_H(Nx,Ny, hoppint_term = 1):
    N = Nx * Ny
    H = np.zeros((N,N), dtype= complex)
    for x in range(Nx):
        for y in range(Ny):
            # cite_index = 2 * (Ny * x + y ) + sublattice
            cite_A_index = Ny * x + y
            cite_B_index = Ny * x + (y + 1) % Ny
            H[cite_B_index,cite_A_index] = hoppint_term
            H[cite_A_index,cite_B_index] = hoppint_term
            cite_B_index = Ny * ((x + 1) % Nx) + y
            H[cite_B_index,cite_A_index] = hoppint_term
            H[cite_A_index,cite_B_index] = hoppint_term
    return H

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

def theta1(z, tau, N=100):
    """
    Calculate the Jacobi theta function θ₁(z, τ)
    
    Parameters:
    z (complex): The first complex variable
    tau (complex): The second complex variable (Im(τ) > 0)
    N (int): Number of terms to sum in the series (default: 100)
    
    Returns:
    complex: The value of θ₁(z, τ)
    """
    q = np.exp(np.pi * 1j * tau)
    n = np.arange(-N, N+1, dtype=complex)
    
    # Calculate the series
    series = np.sum((-1)**n * q**(n*(n+1)/2) * np.exp(2j * np.pi * n * z))
    
    # Multiply by the prefactor
    prefactor = 1j * q**(1/8)
    
    return prefactor * series


def jacobi_theta1(z, tau, N = 100):
    q = np.exp(1j * np.pi * tau)
    n = np.arange(N)  # Adjust the number of terms for accuracy
    return 2 * q**(1/4) * np.sum((-1)**n * q**(n*(n+1)) * np.sin((2*n+1)*z))


def theta_function(z,tau,a,b,k_cutoff = 100):
    k = np.arange(start=-k_cutoff,stop=k_cutoff + 1, dtype= complex)
    result = np.exp(1j *np.pi * tau * (k + a)**2 ) * np.exp(1j * 2 * np.pi * (k + a) * (z + b))
    result = np.sum(result)
    return result

# given a state @state and its @mps of the compact Hilbert space (size of N Choose n) calculate the new state in that space with flux attached.
def flux_attch_on_torus_2_compact_state(state, mps, Nx, Ny):
    for index in range(len(state)):
        flux_factor = 1
        perm = mps.index_2_perm(index)

        # calculte the position of each 2 electrons a>b in complex plane and the phase between them
        for a in range(1 , len(perm)):
            for b in range(a):
                za = cite_index_2_z(perm[a], mps, Ny)
                zb = cite_index_2_z(perm[b], mps, Ny)
                term = theta_function(z=(zb - za) / Nx, tau = 1j * Ny / Nx, a= 1/2, b = 1/2, k_cutoff=20) ** (-2)
                # term = theta1(z = (zb - za) / Nx, tau = 1j * Ny / Nx, N=50) ** 2
                # term = jacobi_theta1(z = (zb - za) / Nx, tau = 1j * Ny / Nx, N=100) ** 2
                if np.abs(term) > 1e-6:
                    flux_factor *= term / np.abs(term)
                    # flux_factor *= term 
        # state[index] *= np.exp(1j * np.angle(flux_factor))
        state[index] *= flux_factor
    return state
    # return normalize(state)
## theta function test

#%%
# Hofstadter_butterfly(Nx = 10, Ny = 10, q = 100)
#%%
# eigen_value_test(Nx=24,Ny=24,p=1,q=3, model = 'chern')

#%%
Nx = 36
Ny = 36
p = -1
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
Ny = 6
p = -1
q = 3

# electron number fill one 'Landau' level
n = Nx * Ny // q

# H_real_space = build_H(Nx,Ny)
H_real_space = build_H(Nx,Ny)
H_real_space_magnetic = add_magnetic_field(np.array(H_real_space), p, q, Nx, Ny, cites_per_uc = 2)

print("---1---")

state, mps = create_IQH_in_extendend_lattice(Nx,Ny,n,extention_factor = 1, band_energy = 1, H_sb = H_real_space_magnetic)

print("---2---")


# state = flux_attch_2_compact_state(np.array(state),mps,Ny)
state = flux_attch_on_torus_2_compact_state(np.array(state),mps,Nx,Ny)

print("---3---")


print_mp_state(state,Nx,Ny,mps)


NN = []
for x in range(Nx):
    for y in range(Ny):
        n1 = cite_2_cite_index(x=x, y=y, sublattice=0, Ny=Ny)
        for i in [0,1]:
            for j in [0,1]:
                n2 = cite_2_cite_index(x=(x - i) % Nx, y=(y - j) % Ny, sublattice=1, Ny=Ny)
                NN.append((n1,n2))

# calculting the energy on the interaction with out mangetic field many body H
# <psi|H_many_body|psi> / <psi|psi>
E = np.matmul(state.T.conjugate(), mps.H_manby_body(H_real_space,state, interaction_strength=0.1, NN = NN)) / np.linalg.norm(state)**2
print(E.real)


