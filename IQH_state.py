#%% 
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft



#%% Auxiliry

# calculate the parity of a given permution @perm and reutrn the parity. 
# @return_sorted_array if True, return the sorted permutaion.
def permutation_parity(perm, return_sorted_array = False):
    inversions = 0
    perm = np.array(perm)
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inversions += 1
                if return_sorted_array:
                    perm[i], perm[j] = perm[j], perm[i]
    if return_sorted_array:
        return (inversions % 2), perm
    else:
        return (inversions % 2)


# Create the multi particle state vector for @N sites and @n particles.
# Reutrn a zero np.vecotr and a dictionary for translating permutation to the correct index in the vector
def multi_particle_state(N, n):
    objects = np.arange(N)
    perms = list((combinations(objects, n)))
    # perms = [np.array(p) for p in perms]
    index = range(len(perms))
    perm_2_index_dict =  {k: o for k, o in zip(perms, index)}
    multi_particle_state_vector = np.zeros((len(perms))) * 1j
    return multi_particle_state_vector, perm_2_index_dict

# # Convert to a NumPy array
# perms_array = np.array(perms)
# # Define N and n
# N = 5
# n = 3

# # Create a list of objects from 0 to N-1
# objects = np.arange(N)

# # Generate all ordered permutations of length n
# perms = list(permutations(objects, n))

# # Convert to a NumPy array
# perms_array = np.array(perms)

# print(perms_array)


#%% single electron Hamiltonian

# parametrs of the model
Nx = 4
Ny = 3

N = Nx * Ny
M = 0
phi = np.pi/4
t1 = 1
t2 = (2-np.sqrt(2))/2

# Building the single particle hamiltonian (h2)
# need to check if the gauge transformation is needed to adress
Kx = np.linspace(-np.pi,np.pi,num=Nx,endpoint=True)
Ky = np.linspace(-np.pi,np.pi,num=Ny,endpoint=True)
X = np.array(range(Nx)) 
Y = np.array(range(Ny)) 

# Kx, Ky = np.meshgrid(kx,ky)
def build_h2(kx, ky):
    h11 = 2 * t2 * (np.cos(kx) - np.cos(ky)) + M
    h12 = t1 * np.exp(1j * phi) * (1 + np.exp(1j * (ky - kx))) + t1 * t1 * np.exp(-1j * phi) * (np.exp(1j * (ky)) + np.exp(1j * (- kx)))
    h2 = np.matrix([[h11, h12], [np.conjugate(h12), -h11]])
    return h2

# eigen states (projected on the lower energies) tensor (state index, A/B lattice, real space position x, real space position y)
# eigen_states = np.zeros((N,2, Nx, Ny)) * 1j

H_k_list = []
H_real_space = np.zeros(shape = (2 * N, 2 * N), dtype = complex) 
state_index = 0
for kx in Kx:
    for ky in Ky:
        H_single_particle = build_h2(kx,ky)
        eig_val, eig_vec = np.linalg.eigh(H_single_particle)

        h_flat = H_single_particle / np.abs(eig_val[0])  # flat band limit
        H_k_list.append(h_flat)
        
        # for x1 in X:
        #     for x2 in X:
        #         for y1 in Y:
        #             for y2 in Y:
        #                 R1_index = 2 * (Ny * x1 + y1)
        #                 R2_index = 2 * (Ny * x2 + y2)
        #                 H_real_space[R1_index : R1_index + 2, R2_index : R2_index + 2] += \
        #                 np.eye(2) * np.exp(-1j * kx * (x1 - x2)) * np.exp(-1j * ky * (y1 - y2))



        # k_space = np.zeros((Nx,Ny))
        # k_space[kx == Kx, ky == Ky] = 1
        # real_space = np.fft.fft2(k_space)

        # eigen_states[state_index, 0, :,:] = eig_vec[0,0] * real_space
        # eigen_states[state_index, 1, :,:] = eig_vec[1,0] * real_space

        # state_index += 1
    
# reshaping each state to one long vector
# eigen_states = np.reshape(eigen_states,(N, 2 * N))

# creaing a block diagonal H_k matrix and dft to real space

H_k = block_diag(*H_k_list)
dft_matrix = np.kron(dft(Nx),(np.kron(dft(Ny),np.eye(2)))) / np.sqrt(N)
H_real_space =np.matmul(np.matmul(dft_matrix.conjugate(),H_k), dft_matrix)

plt.matshow(np.abs(H_real_space))
eig_val, eig_vec = np.linalg.eigh(H_real_space)
# eigen states (projected on the lower energies) tensor (state index, real space position with A,B sublattices)
eigen_states = eig_vec[:,:N].T
print(eig_val)


def normalize(vec):
    return vec/np.linalg.norm(vec)


#%%
# creating the multi particle integer quantum hall state with 2 * N sites and N electron (half filled)
sites = np.arange(2 * N)

# Generate all ordered permutations of length n
perms_array = np.array(list(permutations(sites, N)))
# print(perms_array)

multi_particle_state_vector, perm_2_index_dict = multi_particle_state(2 * N, N)

for perm in perms_array:
    state_coeff = 1
    for electron_index in range(N):
        state_coeff *= eigen_states[electron_index, perm[electron_index]]
    
    parity, soreted_perm = permutation_parity(perm = perm, return_sorted_array = True)
    state_coeff *= (-1)**parity
    multi_particle_state_index = perm_2_index_dict[tuple(soreted_perm)]
    multi_particle_state_vector[multi_particle_state_index] = state_coeff


multi_particle_state_vector = multi_particle_state_vector/np.linalg.norm(multi_particle_state_vector)
print(multi_particle_state_vector)


#%%
# # TODO
# 1. Write a non-interacting many body hamiltonian and check if multi_particle_state_vector is indeed eigen state with the proper energy
# 2. Write a code that creat this state on Qiskit, first with trivial coding (all many body state of the hilbert space) then with constraied hilbert space