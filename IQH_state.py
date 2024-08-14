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


# Create the multi particle state zero vector for @N sites and @n particles.
# Reutrn a zero np.vecotr, a dictionary for translating permutation to the correct index in the vector and a list to translate index to permutation
def multi_particle_state(N, n):
    objects = np.arange(N)
    perms = list((combinations(objects, n)))
    index_2_perm_list = perms
    # perms = [np.array(p) for p in perms]
    index = range(len(perms))
    perm_2_index_dict =  {k: o for k, o in zip(perms, index)}
    multi_particle_state_vector = np.zeros((len(perms))) * 1j
    return multi_particle_state_vector, perm_2_index_dict, index_2_perm_list

# Creat a multi particle state from the given single particle state vectros
# @state_array - np.array of the given states of shape (n,N) where n is the number of states (number of particles to be created)
# and N is the number of sites in the system
def create_multi_particle_state(state_array):
    n,N = np.shape(state_array)
    sites = np.arange(N)

    # Generate all ordered permutations of length n
    perms_array = np.array(list(permutations(sites, N)))
    # print(perms_array)

    multi_particle_state_vector, perm_2_index_dict, index_2_perm = multi_particle_state(N, n)

    for perm in perms_array:
        state_coeff = 1
        for electron_index in range(n):
            state_coeff *= state_array[electron_index, perm[electron_index]]
        
        parity, soreted_perm = permutation_parity(perm = perm, return_sorted_array = True)
        state_coeff *= (-1)**parity
        multi_particle_state_index = perm_2_index_dict[tuple(soreted_perm)]
        multi_particle_state_vector[multi_particle_state_index] = state_coeff


    multi_particle_state_vector = multi_particle_state_vector/np.linalg.norm(multi_particle_state_vector)    
    return multi_particle_state_vector



#%% single electron Hamiltonian

# parametrs of the model
Nx = 2
Ny = 2

N = Nx * Ny
M = 0
phi = np.pi/4
t1 = 1
t2 = (2-np.sqrt(2))/2

# Building the single particle hamiltonian (h2)
# need to check if the gauge transformation is needed to adress (Natanel said no)
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

H_k_list = []
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

    

# creaing a block diagonal H_k matrix and dft to real space

H_k = block_diag(*H_k_list)
# dft matrix as a tensor protuct of dft in x and y axis and idenity in the sublattice
dft_matrix = np.kron(dft(Nx),(np.kron(dft(Ny),np.eye(2)))) / np.sqrt(N)
H_real_space =np.matmul(np.matmul(dft_matrix.conjugate(),H_k), dft_matrix)

eig_val, eig_vec = np.linalg.eigh(H_real_space)

# eigen states (projected on the lower energies) tensor (state index, real space position with A,B sublattices. 
# for position x,y sublattice A the index is 2 * (Ny * x + y) + A
eigen_states = eig_vec[:,:N].T

# creating the multi particle integer quantum hall state with 2 * N sites and N electron (half filled)
multi_particle_state_vector = create_multi_particle_state(eigen_states)

print(multi_particle_state_vector)


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