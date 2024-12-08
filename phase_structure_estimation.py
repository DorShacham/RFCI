#%%
from exact_diagnolization import *
from  IQH_state import *
from chern_ins_magnetic import add_magnetic_field
from scipy import sparse
import matplotlib.pyplot as plt




Nx = 2
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

IQH_state, mps = create_IQH_in_extendend_lattice(Nx,Ny,n,extention_factor = 1, band_energy = 1, H_sb = H_real_space_magnetic)

print("---2----")
# eigenvalues, eigenvectors = exact_diagnolization(Nx, Ny, n=n, band_energy=band_energy, interaction_strength=interaction_strength,k=3,multi_process=False, save_result=False, show_result=False)

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
print_mp_state(IQH_state,Nx,Ny,mps)


for FQH in eigenvectors.T:
    print_mp_state(FQH,Nx,Ny,mps)
    delta_phase = np.angle(IQH_state / FQH)
    sol = sparse.linalg.lsqr(A = sparse_matrix, b= delta_phase,atol=0,btol=0,conlim=1e8,show=True)
    x = sol[0]
    # reidue = np.exp(1j * (sparse_matrix @ x))  - np.exp(1j * delta_phase )
    reidue = (((sparse_matrix @ x)  - delta_phase ) % (2 * np.pi)) / np.pi
    plt.figure()
    plt.hist(reidue)
    print(np.mean(np.abs(reidue)))
