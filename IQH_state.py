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

# Class that deals with with multi_particle_state with @N sites and @n electrons
class Multi_particle_state:
    def __init__(self, N, n):
        objects = np.arange(N)
        perms = list((combinations(objects, n)))
        index_2_perm_list = perms
        index = range(len(perms))
        perm_2_index_dict =  {k: o for k, o in zip(perms, index)}
        self.N = N
        self.n = n
        self.perm_2_index_dict = perm_2_index_dict
        self.index_2_perm_list = index_2_perm_list

# Creats a vectors of zeros in the size of a multi particle state
    def zero_vector(self):
        multi_particle_state_vector = np.zeros((len(self.index_2_perm_list)), dtype = complex)
        return multi_particle_state_vector

# Tranlate index to permutation
    def index_2_perm(self, index):
        return self.index_2_perm_list[index]

# Translate permutation to index in the multi paticle state vector
    def perm_2_index(self, perm):
        return self.perm_2_index_dict[tuple(perm)]

# Create a multi particle state with n electrons and N sites. 
# @state_array is an np.array of shape (n,N) the first index is the electron index.
    def create(self, state_array):
        N = self.N
        n = self.n
        assert((n,N) == np.shape(state_array))
        sites = np.arange(N)

        # Generate all ordered permutations of length n
        perms_array = np.array(list(permutations(sites, n)))
        multi_particle_state_vector = self.zero_vector()

        for perm in perms_array:
            state_coeff = 1
            for electron_index in range(n):
                state_coeff *= state_array[electron_index, perm[electron_index]]
            
            parity, soreted_perm = permutation_parity(perm = perm, return_sorted_array = True)
            state_coeff *= (-1)**parity
            multi_particle_state_index = self.perm_2_index(soreted_perm)
            multi_particle_state_vector[multi_particle_state_index] = state_coeff


        # normalize the state 
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
mps = Multi_particle_state(2 * N, N)
multi_particle_state_vector = mps.create(eigen_states)

print(multi_particle_state_vector)


#%%
# # TODO
# 1. Write a non-interacting many body hamiltonian and check if multi_particle_state_vector is indeed eigen state with the proper energy
# 2. Write a code that creat this state on Qiskit, first with trivial coding (all many body state of the hilbert space) then with constraied hilbert space