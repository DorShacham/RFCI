#%%

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from mpmath import *

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

def add_magnetic_field_square_lattice(H_real_space, p, q, Nx, Ny, cites_per_uc):
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

def add_magnetic_field_chern(H_real_space, p, q, Nx, Ny):
# adding vector potential A = (By,0,0) 
    # sublattice location is (0.5,-0.5)
    cites_per_uc = 2
    x_pos = lambda x, sublattice: 1 * (x + 0 * sublattice / 2) 
    y_pos = lambda y, sublattice: 1 * (y - 0 * sublattice / 2)
    
    H_real_space_magnetic = np.array(H_real_space)
    for x1 in range(Nx):
        for y1 in range(Ny):
            for sublattice1 in range(cites_per_uc):
                cite_A_index = cite_2_cite_index(x=x1,y=y1, sublattice=sublattice1,Ny=Ny)
                for cite_B_index,t in enumerate(H_real_space[:,cite_A_index]):
                    x2, y2, sublattice2 = cite_index_2_cite(cite_B_index, Ny)
                    mean_y =   ( y_pos(y2,sublattice2) + y_pos(y1,sublattice1) ) / 2

                    delta_x_array = np.array([x_pos(x2 ,sublattice2), x_pos(x2 + Nx ,sublattice2), x_pos(x2 - Nx ,sublattice2)]) - x_pos(x1 , sublattice1) 
                    delta_x =   np.min(np.abs(delta_x_array)) * np.sign(delta_x_array[np.argmin(np.abs(delta_x_array))])
                    if not np.array_equal(np.sort(np.abs(delta_x_array)), np.sort(np.unique(np.abs(delta_x_array)))) and np.abs(delta_x) >1e-6:
                        # in case of a symmetry
                        delta_x = 0
                            

                    hopping_phase = np.exp(1j * 2 * np.pi * 2 *  (p / q) * ( delta_x * mean_y))
                    H_real_space_magnetic[cite_B_index,cite_A_index] *= hopping_phase
    return H_real_space_magnetic

def magnetic_FT(H_real_space,Nx,Ny,q,cites_per_uc):
    N = Nx * Ny
    dft_matrix = np.kron(dft(Nx, scale='sqrtn'),(np.kron(dft((Ny // q), scale='sqrtn'),np.eye(cites_per_uc * q))))
    H_k_space =  dft_matrix.T.conjugate() @ H_real_space @ dft_matrix
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
        H_real_space = add_magnetic_field_square_lattice(np.array(H_real_space), p, q, Nx, Ny, cites_per_uc)

    elif model == 'chern':
        cites_per_uc = 2
        H_real_space = build_H(Nx,Ny)
        H_real_space = add_magnetic_field_chern(np.array(H_real_space), p, q, Nx, Ny)


    print(np.sum(np.abs(H_real_space.T.conjugate() - H_real_space)))


    eig_val_original, eig_vec = np.linalg.eigh(H_real_space)

    H_k_space = magnetic_FT(H_real_space, Nx, Ny, q, cites_per_uc=cites_per_uc)
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

def plot_BZ(Nx, Ny, p, q, model = 'basic', flat_band=False):
    if model == 'basic':
        cites_per_uc = 1
        H_real_space = build_basic_H(Nx,Ny)
        H_real_space = add_magnetic_field_square_lattice(np.array(H_real_space),p,q,Nx,Ny,cites_per_uc)
    elif model == 'chern':
        cites_per_uc = 2
        H_real_space = build_H(Nx,Ny, flat_band=flat_band)
        H_real_space = add_magnetic_field_chern(np.array(H_real_space),p,q,Nx,Ny)
    
    H_k_space = magnetic_FT(H_real_space, Nx, Ny, q, cites_per_uc)
    plt.matshow(np.abs(H_k_space[:10,:10]))

    Kx = np.linspace(0, 2 * np.pi,num=Nx,endpoint=False)
    Ky = np.linspace(0, 2 * np.pi / q,num=Ny // q,endpoint=False)
    
    E = np.zeros((cites_per_uc * q, Nx, Ny // q))
    for kx in range(Nx):
        for ky in range(Ny // q):
            unit_cell = H_k_space[(cites_per_uc * q) * ((Ny // q) * kx + ky) : (cites_per_uc * q) * ((Ny // q) * kx + ky + 1),(cites_per_uc * q) * ((Ny // q) * kx + ky) : (cites_per_uc * q) * ((Ny // q) * kx + ky + 1)]
    
            eig_val, eig_vec = np.linalg.eigh(unit_cell)
            E[:,kx,ky] = eig_val


    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    kx, ky = np.meshgrid(Kx, Ky,indexing="ij")

    for r in range(cites_per_uc * q):
        color = plt.cm.tab10(r)  # Tab10 colormap
        ax.plot_surface(kx, ky, E[r,:,:], color=color)

    ax.view_init(elev=5, azim=95)  # Elevation = 10 degrees, Azimuth = 120 degrees
        # Labels and title
    ax.set_xlabel('kx', fontdict={'fontsize': 14})
    ax.set_ylabel('ky', fontdict={'fontsize': 14})
    ax.set_zlabel(r'$E(k_x, k_y)$', fontdict={'fontsize': 14})  # Add labelpad for extra spacing
    # ax.set_title('E(kx, ky)')

    # ax.set_xlim(-np.pi, np.pi)
    # ax.set_ylim(-np.pi/3, np.pi/3)

    # Set tick positions and labels
    xticks = [Kx[0], Kx[Nx//2], Kx[-1]]
    yticks = [Ky[0], Ky[Ny//(2*q)], Ky[-1]]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks([-1, 0, 1])
    ax.set_xticklabels([r"-$\pi$",  r"$0$", r"$\pi$"])
    ax.set_yticklabels([r"-$\pi$",  r"$0$", r"$\pi$"])
    # ax.set_yticklabels([r"-$\frac{\pi}{3}$", r"$0$", r"$\frac{\pi}{3}$"])
    ax.set_zticklabels([r"$-1$", r"$0$", r"$1$"])

    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent clipping
    # plt.show()
    plt.savefig("./BZ.pdf", dpi=300, bbox_inches='tight')
    # plt.savefig("./BZ_magnetic.pdf", dpi=300, bbox_inches='tight')
    return E

def plot_BZ_original_energy(Nx, Ny, p, q):
    cites_per_uc = 2
    H_real_space = build_H(Nx,Ny, flat_band=False)
    H_flat = (build_H(Nx,Ny, flat_band=True)+ np.eye(2 * Nx * Ny)) / 2
    H_real_space = add_magnetic_field_chern(np.array(H_real_space),p,q,Nx,Ny)
    
    N = Nx * Ny
    dft_matrix = np.kron(dft(Nx, scale='sqrtn'),(np.kron(dft((Ny // q), scale='sqrtn'),np.eye(cites_per_uc * q))))
    
    H_k_space = magnetic_FT(H_real_space, Nx, Ny, q, cites_per_uc)

    Kx = np.linspace(0, 2 * np.pi,num=Nx,endpoint=False)
    Ky = np.linspace(0, 2 * np.pi / q,num=Ny // q,endpoint=False)
    
    E = np.zeros((cites_per_uc * q, Nx, Ny // q), dtype=np.complex128)
    eig_vec_real_space = np.zeros_like(H_k_space, dtype=np.complex128)
    for kx in range(Nx):
        for ky in range(Ny // q):
            unit_cell = H_k_space[(cites_per_uc * q) * ((Ny // q) * kx + ky) : (cites_per_uc * q) * ((Ny // q) * kx + ky + 1),(cites_per_uc * q) * ((Ny // q) * kx + ky) : (cites_per_uc * q) * ((Ny // q) * kx + ky + 1)]
            eig_val, eig_vec = np.linalg.eigh(unit_cell)
            eig_vec_real_space[(cites_per_uc * q) * ((Ny // q) * kx + ky) : (cites_per_uc * q) * ((Ny // q) * kx + ky + 1),(cites_per_uc * q) * ((Ny // q) * kx + ky) : (cites_per_uc * q) * ((Ny // q) * kx + ky + 1)] = eig_vec 
    
    eig_vec_real_space = dft_matrix @ eig_vec_real_space
    original_E = eig_vec_real_space.T.conjugate() @ H_flat @ eig_vec_real_space
    for kx in range(Nx):
        for ky in range(Ny // q):
            unit_cell = original_E[(cites_per_uc * q) * ((Ny // q) * kx + ky) : (cites_per_uc * q) * ((Ny // q) * kx + ky + 1),(cites_per_uc * q) * ((Ny // q) * kx + ky) : (cites_per_uc * q) * ((Ny // q) * kx + ky + 1)]
            eig_val = np.diag(unit_cell,k=0)
            E[:,kx,ky] = eig_val

    for r in range(cites_per_uc * q):
        plt.figure()
        plt.imshow(E[r,:,:].real, cmap='viridis', interpolation='nearest')
    # Plot the surface
        plt.colorbar(label="E_flat")
        plt.title(f"band# {r}")

        # Customize axes labels
        # plt.xticks(ticks=np.arange(10), labels=[f"Col {i}" for i in range(10)])
        # plt.yticks(ticks=np.arange(10), labels=[f"Row {i}" for i in range(10)])

        # Show the plot
        # plt.show()
    return E    

def chern_number(Nx,Ny,p,q, model = 'basic'):
    if model == 'basic':
        cites_per_uc = 1
        H_real_space = build_basic_H(Nx,Ny)
        H_real_space = add_magnetic_field_square_lattice(np.array(H_real_space), p, q, Nx, Ny, cites_per_uc)

    elif model == 'chern':
        cites_per_uc = 2
        H_real_space = build_H(Nx,Ny, flat_band=False)
        H_real_space = add_magnetic_field_chern(np.array(H_real_space), p, q, Nx, Ny)



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

def print_band_and_C(Nx,Ny,p,q,model='chern'):

    E = plot_BZ(Nx, Ny, p,q,model)
    C = chern_number(Nx, Ny, p = p,q = q,model = model)
    print(C)
    for n in range(len(C)):
        print(f"C_{n} = {C[n]}")



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
                z = (zb - za) / Nx
                tau = 1j * Ny / Nx
                q = np.exp(1j * np.pi * tau)
                term = jtheta(1,z,q)**2
                if np.abs(term) > 1e-6:
                    flux_factor *= term / np.abs(term)
                    # flux_factor *= term 
        # state[index] *= np.exp(1j * np.angle(flux_factor))
        state[index] *= flux_factor
    return state
    # return normalize(state)
## theta function test

