#%%
# import os
# os.environ["OPENBLAS_NUM_THREADS"] = "10"
# os.environ["MKL_NUM_THREADS"] = "10"
# os.environ["NUMEXPR_NUM_THREADS"] = "10"

# import numpy as np

from exact_diagnolization import *
from  IQH_state import *
from chern_ins_magnetic import add_magnetic_field
from scipy import sparse
import matplotlib.pyplot as plt




Nx = 3
Ny = 6
p = -1
q = 3
band_energy = 1
interaction_strength = 0.1

# electron number fill one 'Landau' level
n = Nx * Ny // q
# n = 4

# H_real_space = build_H(Nx,Ny)
H_real_space = build_H(Nx,Ny)
H_real_space_magnetic = add_magnetic_field(np.array(H_real_space), p, q, Nx, Ny, cites_per_uc = 2)

print("---1---")
try:
    IQH_state = np.load(f'data/Nx-{Nx}_Ny-{Ny}_q=3_magnetic.npy')
    mps = Multi_particle_state(N = 2 * Nx * Ny,n=n)
except:
    print("Calculting IQH state")
    IQH_state, mps = create_IQH_in_extendend_lattice(Nx,Ny,n,extention_factor = 1, band_energy = 1, H_sb = H_real_space_magnetic)
    np.save(f'data/Nx-{Nx}_Ny-{Ny}_q=3_magnetic',IQH_state)

print("---2----")
try:
    loaded = np.load(f'data/Nx-{Nx}_Ny-{Ny}_k-4.npz')
    eigenvectors = loaded['a']
except:
    print("Calculting FQH states")
    eigenvalues, eigenvectors = exact_diagnolization(Nx, Ny, n=n, band_energy=band_energy, interaction_strength=interaction_strength,k=4,multi_process=False, save_result=False, show_result=False)


print("---3----")
# building phase addition matrix
cite_number = 2 * Nx * Ny
mps_2_particles = Multi_particle_state(N=cite_number,n=2)
matrix_elements = {}
for index in range(len(IQH_state)):
    state_perm = mps.index_2_perm(index)
    for cite1 in range(1, len(state_perm)):
        for cite2 in range(cite1):
            matrix_elements[(index, mps_2_particles.perm_2_index((state_perm[cite2],state_perm[cite1])))] = 1
            # A[index, mps_2_particles.perm_2_index((state_perm[cite2],state_perm[cite1]))] = 1

rows, cols = zip(*matrix_elements.keys())
values = list(matrix_elements.values())
sparse_matrix = sparse.csr_matrix((values, (rows, cols)))
# print_mp_state(IQH_state,Nx,Ny,mps)

#%%
H_many_body = np.load_npz(f'results/Exact_Diagnolization/Nx-{Nx}_Ny-{Ny}/sparse_matrix.npz')

for FQH in eigenvectors.T:
    # print_mp_state(FQH,Nx,Ny,mps)
    delta_phase = np.angle(IQH_state / FQH)
    sol = sparse.linalg.lsqr(A = sparse_matrix, b= delta_phase,atol=0,btol=0,conlim=0,show=False)
    x = sol[0]
    # reidue = np.exp(1j * (sparse_matrix @ x))  - np.exp(1j * delta_phase )
    reidue = (((sparse_matrix @ x)  - delta_phase ) ) / np.pi
    plt.figure()
    plt.hist(reidue,bins=100)
    # print(np.mean((reidue)))

    # calculting the energy on the interaction with out mangetic field many body H
    # <psi|H_many_body|psi> / <psi|psi>
    state = np.exp(1j *(sparse_matrix @ x)) * IQH_state
    E = state.T.conjugate() @ (H_many_body @ state) / np.linalg.norm(state)**2
    print(E.real)