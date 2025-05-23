#%%
# import os
# os.environ["OPENBLAS_NUM_THREADS"] = "10"
# os.environ["MKL_NUM_THREADS"] = "10"
# os.environ["NUMEXPR_NUM_THREADS"] = "10"

# import numpy as np

from exact_diagnolization import *
from  IQH_state import *
from chern_ins_magnetic import add_magnetic_field_chern
from scipy import sparse
import matplotlib.pyplot as plt




Nx = 5
Ny = 3
p = -1
q = 3
band_energy = 1
interaction_strength = 0.1

# electron number fill one 'Landau' level
n = Nx * Ny // q
# n = 4

# H_real_space = build_H(Nx,Ny)
H_real_space = build_H(Nx,Ny)
H_real_space_magnetic = add_magnetic_field_chern(np.array(H_real_space), p, q, Nx, Ny)

print("---1---")
try:
    IQH_state = np.load(f'data/Nx-{Nx}_Ny-{Ny}_q=3_magnetic.npy')
    mps = Multi_particle_state(N = 2 * Nx * Ny,n=n)
except:
    print("Calculting IQH state")
    IQH_state, mps = create_IQH_in_extendend_lattice(Nx,Ny,n,extention_factor = 1, band_energy = 1, H_sb = H_real_space_magnetic)
    np.save(f'data/states/Nx-{Nx}_Ny-{Ny}_q=3_magnetic',IQH_state)

print("---2----")
try:
    loaded = np.load(f'data/states/Nx-{Nx}_Ny-{Ny}_k-4.npz')
    eigenvectors = loaded['a']
except:
    print("Calculting FQH states")
    eigenvalues, eigenvectors = exact_diagnolization(Nx, Ny, n=n, band_energy=band_energy, interaction_strength=interaction_strength,k=4,multi_process=False, save_result=False, show_result=False)

# print_mp_state(IQH_state,Nx,Ny,mps)
#%%
print("---3----")
# building phase addition matrix
# cite_number = 2 * Nx * Ny
# mps_2_particles = Multi_particle_state(N=cite_number,n=2)
# matrix_elements = {}
# for index in range(len(IQH_state)):
#     state_perm = mps.index_2_perm(index)
#     for cite1 in range(1, len(state_perm)):
#         for cite2 in range(cite1):
#             matrix_elements[(index, mps_2_particles.perm_2_index((state_perm[cite2],state_perm[cite1])))] = 1
#             # A[index, mps_2_particles.perm_2_index((state_perm[cite2],state_perm[cite1]))] = 1

# rows, cols = zip(*matrix_elements.keys())
# values = list(matrix_elements.values())
# sparse_matrix = sparse.csr_matrix((values, (rows, cols)))
# print_mp_state(IQH_state,Nx,Ny,mps)

loaded = np.load(str(f'data/matrix/ansatz/phase_matrix_Nx-{Nx}_Ny-{Ny}.npz'))
data = loaded['data']
indices = loaded['indices']
shape = tuple(loaded['shape'])

# JAX's BCOO stores indices as shape (nnz, ndim), where each row is a coordinate
# SciPy's coo_matrix expects row, col arrays separately
row = indices[:, 0]
col = indices[:, 1]

# Create COO matrix and convert to CSR
sparse_matrix = sparse.coo_matrix((data, (row, col)), shape=shape).tocsr()


H = sparse.load_npz(str(f'data/matrix/H_Nx-{Nx}_Ny-{Ny}.npz'))
interaction = sparse.load_npz(str(f'data/matrix/interactions_Nx-{Nx}_Ny-{Ny}.npz'))
band_energy = 1e3
interaction_strength = 2
H_many_body = band_energy * (H + n * sparse.identity(n = np.shape(H)[0], format='csr'))  + interaction_strength * interaction


#%%
for i, FQH in enumerate(eigenvectors.T):
    # print_mp_state(FQH,Nx,Ny,mps)
    delta_phase = np.angle(IQH_state / FQH)
    sol = sparse.linalg.lsqr(A = sparse_matrix, b= delta_phase,atol=0,btol=0,conlim=0,show=False)
    x = sol[0]
    print(x)
    # reidue = np.exp(1j * (sparse_matrix @ x))  - np.exp(1j * delta_phase )
    reidue = (((sparse_matrix @ x)  - delta_phase ) ) / np.pi
    plt.figure()
    plt.hist(reidue,bins=100)
    plt.title(f"residue for FQH_{i}")

    plt.figure()
    plt.scatter((delta_phase), (sparse_matrix @ x), color='blue', marker='.')

# Customize the plot
    plt.title(f"Scatter Plot for phases {i}")
    plt.xlabel(f"arg(IQH_magnetic / FQH_{i})")
    plt.ylabel(f"Esitimiated phase for FQH_{i}")

    # Show the plot
    plt.show()
    # calculting the energy on the interaction with out mangetic field many body H
    # <psi|H_many_body|psi> / <psi|psi>
    state = np.exp(1j *(sparse_matrix @ x)) * IQH_state
    E = IQH_state.T.conjugate() @ (H_many_body @ IQH_state) / np.linalg.norm(IQH_state)**2
    print(E.real)
    E = state.T.conjugate() @ (H_many_body @ state) / np.linalg.norm(state)**2
    print(E.real)

#%%

#%%
from jax_ansatz import *
import pickle
# plot phase structure matrix
Nx = 5
Ny = 3
N = 2 * Nx * Ny
n = N // 6
mps = Multi_particle_state(2 * Nx * Ny, n)

z_list = []
for i in range(N):
    z = cite_index_2_z(i,mps,Ny)
    z_list.append(z)

x_list = np.array(z_list).real
y_list = np.array(z_list).imag

ansatz = Jax_ansatz(Nx = Nx, Ny = Ny, n=n, local_layers_num=0, flux_gate_true=True, PLLL=False)
x0 = np.array(ansatz.flux_gate.get_inital_params(), dtype=np.float64)
path = str(f'/Users/dor/Documents/technion/Master/research/code/RFCI/results/vqe_simulation/jax/Nx-5_Ny-3_p--1_q-3/zhzxzfp4/optimization_0/res.pickle')
with open(path, 'rb') as file:
#     # Load the pickled data
    res = pickle.load(file)
# x0 = np.array(res.params[:30], dtype=np.float64)
value_list = x0


# Create scatter plot
plt.figure(figsize=(6,6))
sc = plt.scatter(x_list, y_list, c=value_list, cmap='viridis', s=100)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Complex Numbers Colored by Value')
plt.colorbar(sc, label='Value')
plt.axis('equal')
plt.grid(True)
plt.show()