#%%
# import numpy as np
from jax import config
# Enable 64-bit computation
config.update("jax_enable_x64", True)

from jax import numpy as jnp
import math
import matplotlib.pyplot as plt
from itertools import permutations, combinations, product
from scipy.linalg import block_diag,dft,expm
import scipy
from tqdm import tqdm
import numba



from IQH_state import *

# Class that deals with with multi_particle_state with @N sites and @n electrons
# in first quantization(!)
class Multi_particle_state_first_q:
    def __init__(self, N, n):
        objects = range(N)
        perms = list((permutations(objects, n)))
        index_2_perm_list = perms
        index = range(len(perms))
        perm_2_index_dict =  {k: o for k, o in zip(perms, index)}
        self.N = N
        self.n = n
        self.perms = perms
        self.perm_2_index_dict = perm_2_index_dict
        self.index_2_perm_list = index_2_perm_list
        self.len = len(self.index_2_perm_list)
        
# Creats a vectors of zeros in the size of a multi particle state
    def zero_vector(self):
        multi_particle_state_vector = jnp.zeros((len(self.index_2_perm_list)), dtype = complex)
        return multi_particle_state_vector

# Tranlate index to permutation
    def index_2_perm(self, index):
        return self.index_2_perm_list[index]

# Translate permutation to index in the multi paticle state vector
    def perm_2_index(self, perm):
        return self.perm_2_index_dict[tuple(perm)]

# Create a reduced density matrix rho_a from a [second quantization] @state
# of @n electron on @N cites reduced with particle partitioned 
# with @na electrons
def build_rho_a_2(state, N, n, na):
    nb = n - na
    state = jnp.array(state) / jnp.sqrt(math.factorial(n))
    full_length = int(len(state) * math.factorial(n))
    mps_a = Multi_particle_state_first_q(N, na)
    mps_b = Multi_particle_state_first_q(N, nb)
    mps_full = Multi_particle_state(N, n)

    # build C s.t rho_a = (C @ C_dagger): 
    # https://quantumcomputing.stackexchange.com/questions/7099/how-to-find-the-reduced-density-matrix-of-a-four-qubit-system
 
    try:
        indices_list = jnp.load(f'data/array/rho_a_indices/N-{N}-n-{n}-na-{na}.npz')
        row_indices = indices_list['row_indices']
        col_indices = indices_list['col_indices']
        parities = indices_list['parities']
        state_permuted_indices = indices_list['state_permuted_indices']
    except:
    # Precompute permutations for all indices
        # Map permutations to row and column indices
        row_indices = jnp.zeros((full_length), dtype=np.uint32)
        col_indices = jnp.zeros((full_length), dtype=np.uint32)
        parities = jnp.zeros((full_length), dtype=np.int16)
        state_permuted_indices = jnp.zeros((full_length), dtype=np.uint32)
        f = lambda x: (x[0], mps_full.perm_2_index(x[1]))

        for i, perm in tqdm(enumerate(permutations(range(N), n)), total=full_length):
            row_indices[i] = mps_a.perm_2_index(perm[:na])
            col_indices[i] = mps_b.perm_2_index(perm[-nb:])

            parities[i], state_permuted_indices[i] = f(permutation_parity(perm, return_sorted_array=True))
        # row_indices, col_indices = zip(*[(mps_a.perm_2_index(perm[:na]), mps_b.perm_2_index(perm[-nb:])) for perm in permutations(range(N), n)])
        # row_indices = jnp.array(row_indices)
        # col_indices = jnp.array(col_indices)

        # Compute permutation parities in batch
        # f = lambda x: (x[0], mps_full.perm_2_index(x[1]))
        # parities, state_permuted_indices = zip(*[f(permutation_parity(perm, return_sorted_array=True)) for perm in permutations(range(N), n)])
        # parities = jnp.array(parities)
        # state_permuted_indices = jnp.array(state_permuted_indices)
        
        jnp.savez(f'data/array/rho_a_indices/N-{N}-n-{n}-na-{na}.npz',
        row_indices = row_indices,
        col_indices = col_indices,
        parities = parities,
        state_permuted_indices = state_permuted_indices)

    # Construct the matrix C efficiently
    print("Now!")
    if len(np.shape(state)) == 1: # a single vector:
        state = state.reshape((len(state),1)) 
    
    # row_indices = row_indices.reshape((len(row_indices), 1))
    # col_indices = col_indices.reshape((len(col_indices), 1))
    parities = parities.reshape((len(parities),1))
    C = jnp.empty((mps_a.len, mps_b.len, jnp.shape(state)[1]), dtype=np.complex128)
    C = C.at[row_indices, col_indices,:].set(state[state_permuted_indices,:] * (-1) ** parities)
    
    rho = jnp.mean(np.array([C[:,:,i] @ C[:,:,i].T.conjugate() for i in range(np.shape(state)[1])]), axis= 0)   
    return rho 


# Create a reduced density matrix rho_a from a [second quantization] @state
# of @n electron on @N cites reduced with particle partitioned 
# with @na electrons
def build_rho_a(state, N, n, na):
    nb = n - na
    state = jnp.array(state) / jnp.sqrt(math.factorial(n))
    mps = Multi_particle_state_first_q(N, na)
    mps_full = Multi_particle_state(N, n)
    rho = jnp.zeros(shape=(mps.len,mps.len), dtype=np.complex128)

    for row_index in tqdm(range(mps.len)):
        for col_index in range(mps.len):
            row_perm = mps.index_2_perm(row_index)
            col_perm = mps.index_2_perm(col_index)
            
            # tracing out the last nb = n - na electrons:
            perm_b_row = set(permutations([x for x in range(N) if x not in row_perm], nb))
            perm_b_col = set(permutations([x for x in range(N) if x not in col_perm], nb))
            common_elements = perm_b_row & perm_b_col
            if len(common_elements) > 0:
                for perm_b in common_elements:

                    full_row_perm = tuple(list(row_perm) + list(perm_b))
                    row_parity, row_soreted_perm =  permutation_parity(full_row_perm, return_sorted_array=True)
                    row_full_index = mps_full.perm_2_index(row_soreted_perm)
                    
                    full_col_perm = tuple(list(col_perm) + list(perm_b))
                    col_parity, col_soreted_perm =  permutation_parity(full_col_perm, return_sorted_array=True)
                    col_full_index = mps_full.perm_2_index(col_soreted_perm)
                    rho[row_index, col_index] +=  state[row_full_index] * state[col_full_index].conjugate() * (-1) **  (row_parity + col_parity)

    return rho


# compute the entanglement_spectrum of a [second quantization] @state
# of @n electron on @N cites reduced with particle partitioned 
# with @na electrons
def entanglement_spectrum(state, N, n, na):
    rho = build_rho_a_2(state, N, n, na)
    eigval = jnp.linalg.eigvalsh(rho)
    return eigval

# compute the entanglement_entropy of a [second quantization] @state
# of @n electron on @N cites reduced with particle partitioned 
# with @na electrons
def entanglement_entropy(state, N, n, na):
    eigval = entanglement_spectrum(state, N, n, na)
    non_zero_eigval = eigval[np.abs(eigval) > 1e-12]
    S = - jnp.sum(non_zero_eigval * jnp.log(non_zero_eigval))
    return S

if __name__ == "__main__":
    print("Hey")
#%%
# N = 3
# n = 6
# na = 1

# a1 = 1
# a2 = 2
# state = normalize(np.array([a1,0,a2]))
# a1 = state[0]
# a2 = state[2]

# rho = build_rho_a_parallel(state,N,n,na)
# print(rho)

# print("-------------")
# analytic = jnp.array([[np.abs(a1)**2 / 2, 0, -a1 * a2.conjugate() / 2],
#                     [0,                 1/2,  0],
#                     [-a1.conjugate()*a2 / 2, 0,   jnp.abs(a2)**2 / 2]])

# print(analytic)

# print(np.sum(np.abs(rho - rho.T.conjugate())))
# print(np.sum(np.abs(rho - analytic)))


# #%%
# Nx = 2
# Ny = 6

# N = 2 * Nx * Ny
# n = N // 6
# na = 1

# # FF_state, mps =  create_IQH_in_extendend_lattice(Nx,Ny,n,extention_factor = 1, band_energy = 1, H_sb = None)
# # S_F = entanglement_entropy(state = FF_state, N = N, n = n, na = na)
# S_F = jnp.log(math.factorial(n) / math.factorial(na) / math.factorial(n - na))
# print(f"S_F = {S_F}")

# loaded2 = jnp.load(f'data/states/Nx-{Nx}_Ny-{Ny}_k-4.npz')
# eigenvectors = loaded2['a']
# # S_int = entanglement_entropy(state = eigenvectors[:,0], N = N, n = n, na = na)
# # S_int = entanglement_entropy(state = loaded, N = N, n = n, na = na)
# print(f"S_int = {S_int}")

# print(f"S_int - S_F = {S_int - S_F}")
