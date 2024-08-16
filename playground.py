#%%
#%% 
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft



# Auxiliry

# Normalizes @vec
def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    else:
        return vec/norm

def print_sb_state(state,Nx,Ny):
    map = np.zeros((Ny, 2 * Nx), dtype = complex)
    for x in range((Nx)):
        for y in range(Ny):
            map[y,2 * x] = state[2 * (Ny * x + y)]
            map[y,2 * x + 1] = state[2 * (Ny * x + y) + 1]
    plt.matshow(np.abs(map))

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
            multi_particle_state_vector[multi_particle_state_index] += state_coeff


        return normalize(multi_particle_state_vector)

# Function that acts as a (non-interacting) many body Hamiltonian bases on a single body Hamiltonian @H_sb on state @multi_particle_state
# The calculation is taking each unit vector in the multi-particle state, splitting it to single particle state calculating the action of
# the single body hamiltonian on the single particle states and from the new single particle states calculating a new multi-particle state.
# This is done for every unit multi_particle_state vector. Lastly summing their amplitude will result the new multi-particle state.
    def H_manby_body(self, H_sb, multi_particle_state):
        new_state = self.zero_vector()

        for index in range(len(multi_particle_state)):
            signle_body_states = self.index_2_perm(index)
            #signle_body_states is the indices of the sites where the electrons seats.
            #Since they are unit vecotrs the action of H_sb is just taking the apropriate column
            new_single_particle_states = H_sb[:,signle_body_states]
            # Summing the new multi-particle state with the right coeff
            new_state += self.create(new_single_particle_states.T) * multi_particle_state[index]
        
        return new_state



#%% testing 

# parametrs of the model
Nx = 20
Ny = 20

N = Nx * Ny
t1 = 1
sub = 2
# Building the single particle hamiltonian (h2)
# need to check if the gauge transformation is needed to adress (Natanel said no)
Kx = np.linspace(0, 2 * np.pi,num=Nx,endpoint=False) 
Ky = np.linspace(0, 2 * np.pi,num=Ny,endpoint=False) 
X = np.array(range(Nx)) 
Y = np.array(range(Ny)) 

# Kx, Ky = np.meshgrid(kx,ky)
def build_h2(kx, ky):
    return -2 * t1 * (np.cos(kx) + np.cos(ky)) * np.eye(sub)

H_k_list = []
for kx in Kx:
    for ky in Ky:
        H_k_list.append(build_h2(kx,ky))
        
# creaing a block diagonal H_k matrix and dft to real space

H_k = block_diag(*H_k_list)

# dft matrix as a tensor protuct of dft in x and y axis and idenity in the sublattice
dft_matrix = np.kron(dft(Nx),(np.kron(dft(Ny),np.eye(sub)))) / np.sqrt(N)
H_real_space =np.matmul(np.matmul(dft_matrix.T.conjugate(),H_k), dft_matrix)
# eig_val, eig_vec = np.linalg.eigh(H_real_space)

index = lambda x,y,A=0: sub * (Ny * x + y) + A
vectors = np.eye(N * sub)
H_real_space_2 = np.zeros((N * sub, N * sub))
for i in range(Nx):
    for j in range(Ny):
        H_real_space_2 += -t1 * np.outer(vectors[:,index(i,j)], vectors[:,index((i+1)%Nx,j)] + vectors[:,index(i,(j+1)%Ny)] +  vectors[:,index((i-1)%Nx,j)] + vectors[:,index(i,(j-1)%Ny)])
        H_real_space_2 += -t1 * np.outer(vectors[:,index(i,j,1)], vectors[:,index((i+1)%Nx,j,1)] + vectors[:,index(i,(j+1)%Ny,1)] +  vectors[:,index((i-1)%Nx,j,1)] + vectors[:,index(i,(j-1)%Ny,1)])
plt.matshow(np.abs(H_real_space))
plt.matshow(np.abs(H_real_space_2))
print(np.diag(np.matmul(np.matmul(dft_matrix,H_real_space_2), dft_matrix.T.conjugate())).real)
print(H_k_list)
# print(H_real_space_2)
state = np.zeros((N * sub))
x = 1
y = 1
index = sub * (Ny * x + y) 
state[index] =1
# print_sb_state(state,Nx,Ny)
# print_sb_state(H_real_space[:,index],Nx,Ny)