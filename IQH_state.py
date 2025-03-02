#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
import scipy
from tqdm import tqdm

from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Statevector




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
            map[2 * y,2 * x] = np.abs(state[2 * (Ny * x + y)])**2
            map[2 * y + 1,2 * x + 1] = np.abs(state[2 * (Ny * x + y) + 1])**2
    plt.matshow(np.abs(map))


# for a given @state [Multi-particle state] on lattice @Nx,@Ny print a heatmap of the distribution of electrons.
# if @saveto is not None should be a path to save location for the heatmap
def print_mp_state(state,Nx,Ny,mps, saveto = None):
    map = np.zeros((2 * Ny, 2 * Nx), dtype = complex)
    v = np.zeros(mps.N, dtype= complex)
    for index in range(len(state)):
        perm = mps.index_2_perm(index)
        for p in perm:
            v[p] += np.abs(state[index])**2
    for x in range((Nx)):
        for y in range(Ny):
            map[2 * y ,2 * x] += v[2 * (Ny * x + y)]
            map[2 * y + 1,2 * x + 1] += v[2 * (Ny * x + y) + 1]
    
    plt.figure()
    plt.matshow(np.abs(map))
    plt.colorbar()
    if saveto is None:
        plt.show()
    else:
        plt.savefig(saveto)
    plt.close()

def print_state_vector(state_vector,Nx,Ny, saveto = None):
    sv = Statevector(state_vector)
    N = 2 * Nx * Ny
    n_cite = lambda cite_index: SparsePauliOp([str("I"*(N - cite_index - 1) + "Z" + "I"*cite_index)],  [1])
    map = np.zeros((2 * Ny, 2 * Nx), dtype = complex)
    for x in range((Nx)):
        for y in range(Ny):
            map[2 * y,2 * x] = sv.expectation_value(n_cite(2 * (Ny * x + y))) * (-0.5) + 0.5 
            map[2 * y + 1,2 * x + 1] = sv.expectation_value(n_cite(2 * (Ny * x + y) + 1)) * (-0.5) + 0.5 
    plt.figure()
    plt.matshow(np.abs(map))
    plt.colorbar()
    if saveto is None:
        plt.show()
    else:
        plt.savefig(saveto)
    plt.close()



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
        self.len = len(self.index_2_perm_list)
        
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
        perms_array = permutations(sites, n)
        multi_particle_state_vector = self.zero_vector()

        for perm in perms_array:
            state_coeff = 1
            for electron_index in range(n):
                state_coeff *= state_array[electron_index, perm[electron_index]].conjugate()
            
            parity, soreted_perm = permutation_parity(perm = perm, return_sorted_array = True)
            state_coeff *= (-1)**parity
            multi_particle_state_index = self.perm_2_index(soreted_perm)
            multi_particle_state_vector[multi_particle_state_index] += state_coeff

        return normalize(multi_particle_state_vector)

# Function that acts as a (non-interacting) many body Hamiltonian bases on a single body Hamiltonian @H_sb on state @multi_particle_state
# The calculation is taking each unit vector in the multi-particle state, splitting it to single particle state calculating the action of
# the single body hamiltonian on the single particle states and from the new single particle states calculating a new multi-particle state.
# This is done for every unit multi_particle_state vector. Lastly summing their amplitude will result the new multi-particle state.
    def H_manby_body(self, H_sb, multi_particle_state, interaction_strength = 0, NN = None):
        N, M = np.shape(H_sb)
        assert(N == M)
        new_state = self.zero_vector()

        for index in range(len(multi_particle_state)):
            state_perm = self.index_2_perm(index)
            #signle_body_states is the indices of the sites where the electrons seats.
            for i in range(N):
                for j in range(M):
                    if (i == j) and (i in state_perm):
                        new_state[index] += H_sb[i,j].conjugate() * multi_particle_state[index]

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
                            new_state[new_index] += H_sb[i,j].conjugate() * (-1)**k * (-1)**parity * multi_particle_state[index]

            if interaction_strength != 0:
                for i,j in NN:
                    if (i in state_perm) and (j in state_perm):
                        new_state[index] += interaction_strength * multi_particle_state[index]
        
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
def build_H(Nx = 2, Ny = 2, band_energy = 1, M = 0, phi = np.pi/4, phase_shift_x = 0, phase_shift_y = 0, element_cutoff= None):
# parametrs of the model
    N = Nx * Ny
    # phi = np.pi/4
    t1 = -1
    t2 = -(2-np.sqrt(2))/2 * t1
    # t2 = t1 / np.sqrt(2)

    # Building the single particle hamiltonian (h2)
    # need to check if the gauge transformation is needed to adress (Natanel said no)
    # Starting the BZ from zero to 2pi since this is how the DFT matrix is built
    
    Kx = np.linspace(0, 2 * np.pi,num=Nx,endpoint=False)
    Ky = np.linspace(0, 2 * np.pi,num=Ny,endpoint=False)
    # Kx = np.linspace(-np.pi, np.pi,num=Nx,endpoint=False)
    # Ky = np.linspace(-np.pi, np.pi,num=Ny,endpoint=False)
    X = np.array(range(Nx)) 
    Y = np.array(range(Ny)) 

    def build_h2(kx, ky, band_energy):
        h11 = 2 * t2 * (np.cos(kx) - np.cos(ky)) + M
        h12 = t1 * np.exp(1j * phi) * (1 + np.exp(1j * (ky - kx))) + t1 * np.exp(-1j * phi) * (np.exp(1j * (ky)) + np.exp(1j * (- kx)))
        h2 = np.array([[h11, h12], [np.conjugate(h12), -h11]])
        return h2

    H_k_list = []
    i = 0
    for kx in Kx:
        for ky in Ky:
            H_single_particle = build_h2(kx + phase_shift_x/Nx,ky + phase_shift_y/Ny, band_energy)
            eig_val, eig_vec = np.linalg.eigh(H_single_particle)
            h_flat = H_single_particle / np.abs(eig_val[0]) * band_energy + i * 1e-8  # flat band limit + small disperssion for numerical stabilty
            H_k_list.append(h_flat)
            # H_k_list.append(H_single_particle)
            i += 1
            
    # creaing a block diagonal H_k matrix and dft to real space

    H_k = block_diag(*H_k_list)

    # dft matrix as a tensor protuct of dft in x and y axis and idenity in the sublattice
    dft_matrix = np.kron(dft(Nx, scale='sqrtn'),(np.kron(dft(Ny, scale='sqrtn'),np.eye(2))))
    # dft_matrix = np.kron(lattice_dft(Nx),(np.kron(lattice_dft(Ny),np.eye(2))))
    H_real_space = dft_matrix.T.conjugate() @ H_k @ dft_matrix

    if element_cutoff is not None:
        H_real_space[np.abs(H_real_space) < element_cutoff] = 0
    
    return H_real_space

# Creating an Interger Quantum Hall state on a 2 * Nx * Ny lattice and then extend the lattice by extention_factor
# in the x direction
# if H_sb is not None build the state according to it
# return the state vector, extended_mps
def create_IQH_in_extendend_lattice(Nx,Ny,n,extention_factor = 1, band_energy = 1, H_sb = None):
    N = 2 * Nx * Ny
    if H_sb is None:
        H_real_space = build_H(Nx, Ny, band_energy)
    else:
        H_real_space = H_sb
    eig_val, eig_vec = np.linalg.eigh(H_real_space)
# eigen states (projected on the lower energies) tensor (state index, real space position with A,B sublattices. 
# for position x,y sublattice A the index is 2 * (Ny * x + y) + A
    eigen_states = eig_vec[:,:n].T
# creating the multi particle integer quantum hall state with  N sites and n electron (half filled)
    mps = Multi_particle_state(N, n)
    multi_particle_state_vector = mps.create(eigen_states)
# Exatend the system on the x axis by adding unequippied cites. @extention_factor is a positive integer describe 
# how many cites to add i.e N_new = extention_factor * N
    new_N = extention_factor * N
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

# project the @state with @mps on @band = +1/-1 of Hamiltonin @H and return the overlap
def project_on_band(state,band, H, mps, return_k_occupation = False):
    assert(band == 1 or band == -1)
    N = mps.N
    n = mps.n
    eig_val, eig_vec = np.linalg.eigh(H)   
    if band == -1:
        if not return_k_occupation:
            print("Calculting the number of electrons in the lower band:")
        eig_vec = eig_vec[:,: N // 2]
    elif band == 1:
        if not return_k_occupation:
            print("Calculting the number of electrons in the upper band:")
        eig_vec = eig_vec[:,N // 2:]

    k_occupation = []
    # In this caluclation we calculate sum_k(<state|C_k^dagger C_k |state>) = sum_k( |C_k |state>|^2)
    # Thus first we calculate C_k|state> and then it's norm squared. C_k kills an electron, thus we need a state with minus one electron.
    new_mps = Multi_particle_state(N, n - 1)
    
    for k in range(N // 2):
        new_state = new_mps.zero_vector()
        # the state = \Sigma_j=0 ^len(state) \alpha_i C_1_i^dagger...C_N_i^dagger|0>
        for index in range(len(state)):
            perm = mps.index_2_perm(index)
        # Running on the j of the sum. If there is no C_j_dagger we have C_j|0> = 0  
        # If there is C_j_dagger this will turn to another state
            for j in range(N):
                if not j in perm:
                    continue
                else:
                    l = perm.index(j)
                    new_perm = list(perm)
                    del new_perm[l]
                    new_index = new_mps.perm_2_index(new_perm)
                    new_state[new_index] += (-1)**l * state[index] * eig_vec[j,k]
            
    
        k_occupation.append(np.linalg.norm(new_state)**2)
    
    if return_k_occupation:
        return k_occupation
    else:
        band_occupation = np.sum(np.array(k_occupation))
    print(f"--------\nThere are {band_occupation} electrons")
    return band_occupation



# translate the n electrno state with @mps index to the full Hilbert space state vector index
def index_2_state_vector_index(index, mps):
    perm = mps.index_2_perm(index)
    new_index = 0
    for p in perm:
        new_index += 2**p

    return new_index

# translate the index of the full Hilbert state vector to a tuple of the cite inidices of each electron (perm)
def full_index_2_perm(full_index):
    indices = [i for i, bit in enumerate(binary_string[::-1]) if bit == '1']
    return tuple(indices)


# translate the n electrno state with @mps to the full Hilbert space state vector
def state_2_full_state_vector(state, mps):
    N = mps.N
    full_state_vector = np.zeros(shape = 2**N, dtype = complex)

    for index in range(len(state)):
        new_index = index_2_state_vector_index(index,mps)
        full_state_vector[new_index] = state[index]

    return full_state_vector

# translate the @x,@y and basis @sublattice=0/1 to a cite index = 2 * (Ny * x + y ) + sublattice
def cite_2_cite_index(x,y,sublattice, Ny):
    return 2 * (Ny * x + y ) + sublattice

# translate the cite @index = 2 * (Ny * x + y) + sublattice -> (x,y,sublattice)
def cite_index_2_cite(index, Ny):
    sublattice = index % 2 
    y = (index // 2 ) % Ny
    x = index // (2 * Ny)
    return (x,y,sublattice)

# translate the cite @index = 2 * (Ny * x + y) + sublattice -> z = x + iy
def cite_index_2_z(index,mps, Ny):
    x,y,sublattice = cite_index_2_cite(index,Ny)
### maybe should use also the sublattice index

    z = (x  + 0 * sublattice) + 1j * y
    # z =  y + 1j * (x + 0.2 * sublattice)
    return z

def lattice_dft(N):
    dft_row = []
    K = np.linspace(start=-np.pi, stop = np.pi, num = N, endpoint=False, dtype=np.complex128)
    for i in range(N):
        dft_row.append(np.exp(1j * i * K))
    dft_matrix = np.array(dft_row, dtype=np.complex128)
    return 1/np.sqrt(N) * dft_matrix


# Implimintation of the translation operator for @state with @mps on lattice (@Nx,@Ny) in the (@Tx,@Ty) direction
def translation_operator(state, mps, Nx, Ny, Tx = 0, Ty = 0):
    new_state = mps.zero_vector()
    for index in (range(mps.len)):
        state_perm = mps.index_2_perm(index)
        new_perm = []
        for cite_index in state_perm:
            x,y, sublatice = cite_index_2_cite(cite_index,Ny)
            new_cite_index = cite_2_cite_index((x + Tx) % Nx, (y + Ty) % Ny , sublatice, Ny)
            new_perm.append(new_cite_index)
        parity, sorted_perm = permutation_parity(tuple(new_perm), return_sorted_array=True)
        new_index = mps.perm_2_index(sorted_perm)
        new_state[new_index] = (-1) ** parity * state[index]
    return new_state

# Implimintation of the translation operator for @state with @mps on lattice (@Nx,@Ny) in the (@Tx,@Ty) direction
def translation_matrix(mps, Nx, Ny, Tx = 0, Ty = 0):
    col_list = []
    for index in tqdm(range(mps.len)):
        state = mps.zero_vector()
        state[index] += 1
        new_state = translation_operator(state,mps,Nx,Ny,Tx,Ty)
        col_list.append(new_state)
    
    matrix = np.array(col_list).T
    sparse_matrix = scipy.sparse.csr_matrix(matrix)
    return sparse_matrix