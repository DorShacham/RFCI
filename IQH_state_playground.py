#%% 
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
from tqdm import tqdm


#%% Auxiliry

# Normalizes @vec
def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    else:
        return vec/norm

def print_sb_state(state,Nx,Ny):
    map = np.zeros((2 * Ny, 2 * Nx), dtype = complex)
    for x in range((Nx)):
        for y in range(Ny):
            map[2 * y,2 * x] = state[2 * (Ny * x + y)]
            map[2 * y,2 * x + 1] = state[2 * (Ny * x + y) + 1]
    plt.matshow(np.abs(map))

def print_mp_state(state,Nx,Ny,mps):
    map = np.zeros((2 * Ny, 2 * Nx), dtype = complex)
    for index in range(len(state)):
        v = np.zeros(mps.N, dtype= complex)
        perm = mps.index_2_perm(index)
        for p in perm:
            v[p] = state[index]    
        for x in range((Nx)):
            for y in range(Ny):
                map[2 * y,2 * x] += v[2 * (Ny * x + y)]
                map[2 * y,2 * x + 1] += v[2 * (Ny * x + y) + 1]
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
        N, M = np.shape(H_sb)
        assert(N == M)
        new_state = self.zero_vector()

        for index in range(len(multi_particle_state)):
            state_perm = self.index_2_perm(index)
            #signle_body_states is the indices of the sites where the electrons seats.
            for i in range(N):
                for j in range(M):
                    if (i == j):
                        new_state[index] += H_sb[i,j] * multi_particle_state[index] 
                    else:
                        # Ci_dagger * Ci_dagger = C_j |0> = 0 
                        if (i in state_perm) or (not (j in state_perm)):
                            continue 
                        else:
                            # find index of Cj
                            k = state_perm.index(j)
                            # contract Cj with Cj_dagger and insert Ci_dagger to the right place 
                            new_perm =  list(state_perm)
                            del new_perm[k]
                            new_perm.insert(0,i)
                            parity, sorted_perm = permutation_parity(tuple(new_perm), return_sorted_array=True)
                            new_index = self.perm_2_index(sorted_perm)
                
                            # Summing the new multi-particle state with the right coeff
                            new_state[new_index] += H_sb[i,j] * (-1)**k * (-1)**parity * multi_particle_state[index]
        
        return new_state

# Time evolve a multiparticle state for non-interacting Hamltonian
    def time_evolve(self, H_sb, multi_particle_state, t = 1):
        h_bar = 1
        U = expm(-1j * t *  H_sb / h_bar)
        new_state = self.zero_vector()

        for index in tqdm(range(len(multi_particle_state))):
            state_perm = self.index_2_perm(index)
            state_list = []
            for n in state_perm:
                state_list.append(U[:,n])

            new_state += self.create(np.array(state_list)) * multi_particle_state[index]
        return new_state

# Create the single body Hamiltonian in real space
def build_H(Nx = 2, Ny = 2):
# parametrs of the model
    N = Nx * Ny
    M = 0
    phi = np.pi/4
    t1 = 1
    t2 = (2-np.sqrt(2))/2

    # Building the single particle hamiltonian (h2)
    # need to check if the gauge transformation is needed to adress (Natanel said no)
    # Starting the BZ from zero to 2pi since this is how the DFT matrix is built
    Kx = np.linspace(0, 2 * np.pi,num=Nx,endpoint=False)
    Ky = np.linspace(0, 2 * np.pi,num=Ny,endpoint=False)
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
    H_real_space =np.matmul(np.matmul(dft_matrix.T.conjugate(),H_k), dft_matrix)
    return H_real_space

# Creating an Interger Quantum Hall state on a 2 * Nx * Ny lattice and then extend the lattice by extention_factor
# in the x direction
# return the state vector, extended_mps
def create_IQH_in_extendend_lattice(Nx,Ny,extention_factor):
    N = Nx * Ny
    H_real_space = build_H(Nx, Ny)
    eig_val, eig_vec = np.linalg.eigh(H_real_space)
# eigen states (projected on the lower energies) tensor (state index, real space position with A,B sublattices. 
# for position x,y sublattice A the index is 2 * (Ny * x + y) + A
    n = N
    eigen_states = eig_vec[:,:n].T
# creating the multi particle integer quantum hall state with 2 * N sites and n=N electron (half filled)
    mps = Multi_particle_state(2 * N, n)
    multi_particle_state_vector = mps.create(eigen_states)
# Exatend the system on the x axis by adding unequippied cites. @extention_factor is a positive integer describe 
# how many cites to add i.e N_new = extention_factor * N
    new_N = 2 * (extention_factor * Nx) * Ny   
    extended_mps = Multi_particle_state(N = new_N, n = n)
    extended_state = extended_mps.zero_vector()

    state = multi_particle_state_vector

    for index in range(len(state)):
        perm = mps.index_2_perm(index)
        new_perm = []
        for i in range(n):
            cite_index = perm[i]
            x = cite_index // (2 * Ny)
            new_cite_index = cite_index + 2 * Ny * (extention_factor - 1) * x
            new_perm.append(new_cite_index)
        
        assert(permutation_parity(perm,False) == permutation_parity(new_perm,False))
        new_index = extended_mps.perm_2_index(new_perm)
        extended_state[new_index] = state[index]

    return extended_state, extended_mps


# %%
Nx = 2
Ny = 2
N = Nx * Ny

H_real_space = build_H(Nx, Ny)

eig_val, eig_vec = np.linalg.eigh(H_real_space)
# eigen states (projected on the lower energies) tensor (state index, real space position with A,B sublattices. 
# for position x,y sublattice A the index is 2 * (Ny * x + y) + A
n = N
eigen_states = eig_vec[:,:n].T

# creating the multi particle integer quantum hall state with 2 * N sites and n=N electron (half filled)
mps = Multi_particle_state(2 * N, n)
multi_particle_state_vector = mps.create(eigen_states)

new_mp_state = mps.H_manby_body(H_real_space,multi_particle_state_vector)
# sanity check
print((new_mp_state[np.abs(new_mp_state)>1e-8]/multi_particle_state_vector[np.abs(multi_particle_state_vector)>1e-8]).real)
print(np.linalg.norm(normalize(new_mp_state) + multi_particle_state_vector))
print(np.linalg.norm(multi_particle_state_vector))
print(np.linalg.norm(new_mp_state))
print_mp_state(multi_particle_state_vector, Nx = Nx, Ny = Ny, mps = mps)
# print_mp_state(new_mp_state, Nx = Nx, Ny = Ny, mps = mps)
print_mp_state(mps.time_evolve(H_real_space,multi_particle_state_vector,t=100), Nx = Nx, Ny = Ny, mps = mps)

#%% 
# Exatend the system on the x axis by adding unequippied cites. @extention_factor is a positive integer describe 
# how many cites to add i.e N_new = extention_factor * N

extention_factor = 3
new_N = 2 * (extention_factor * Nx) * Ny   
extended_mps = Multi_particle_state(N = new_N, n = n)
extended_state = extended_mps.zero_vector()

state = multi_particle_state_vector

for index in range(len(state)):
    perm = mps.index_2_perm(index)
    new_perm = []
    for i in range(n):
        cite_index = perm[i]
        x = cite_index // (2 * Ny)
        new_cite_index = cite_index + 2 * Ny * (extention_factor - 1) * x
        new_perm.append(new_cite_index)
    
    assert(permutation_parity(perm,False) == permutation_parity(new_perm,False))
    new_index = extended_mps.perm_2_index(new_perm)
    extended_state[new_index] = state[index]

state = create_IQH_in_extendend_lattice(Nx, Ny, extention_factor)
print(np.linalg.norm(extended_state - state))
#%%
extended_H = build_H(Nx = extention_factor * Nx, Ny = Ny)
new_extended_state = extended_mps.H_manby_body(extended_H,extended_state)

# sanity check
# print((new_extended_state[np.abs(new_extended_state)>1e-8]/extended_state[np.abs(extended_state)>1e-8]).real)
print(np.linalg.norm(normalize(new_extended_state) + extended_state))
print(np.linalg.norm(extended_state))
print(np.linalg.norm(new_extended_state))
print_mp_state(extended_state, Nx = extention_factor * Nx, Ny = Ny, mps = extended_mps)
# print_mp_state(prometed_state, Nx = extention_factor * Nx, Ny = Ny, mps = extended_mps)
print_mp_state(extended_mps.time_evolve(extended_H, extended_state, t=0.07), Nx = extention_factor * Nx, Ny = Ny, mps = extended_mps)

#%% 
Nx = 3
Ny = 3
N = 2 * Nx * Ny
n = 3
mps2 = Multi_particle_state(N,n)
H = build_H(Nx,Ny)
states = np.eye(N)[:,:n].T
# tmp = states[0]
# states[0] = states[1]
# states[1] = np.array([1., 0., 0., 0., 0.])
state = mps2.create(states)
print_mp_state(state, Nx = Nx, Ny = Ny, mps = mps2)
print_mp_state(mps2.time_evolve(H_sb=H,multi_particle_state=state,t=0.07), Nx = Nx, Ny = Ny, mps = mps2)
# %%

# %%
#%%
# # TODO
# 1. Write a non-interacting many body hamiltonian and check if multi_particle_state_vector is indeed eigen state with the proper energy
# 2. Write a code that creat this state on Qiskit, first with trivial coding (all many body state of the hilbert space) then with constraied hilbert space