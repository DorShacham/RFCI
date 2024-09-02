#%% 
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
from tqdm import tqdm

from IQH_state import create_IQH_in_extendend_lattice, print_mp_state


#%%

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

# translate the cite @index = 2 * (Ny * x + y) + sublattice -> z = x + iy
def cite_index_2_z(index,mps):
    sublattice = index % 2 
    y = (index // 2 ) % Ny
    x = index // (2 * Ny)
### maybe should use also the subllatice index

    z = x + 1j * y
    return z



#%%
# Preparing the Full Hilbert 2^N state
Nx = 2
Ny = 2
extention_factor = 3

state, mps = create_IQH_in_extendend_lattice(Nx = Nx, Ny = Ny, extention_factor = extention_factor)

Nx = extention_factor * Nx
N = 2 * Nx * Ny

#%%
# Fulx attachment [without qiskit]

# Calcute the accumlated phase for each cell
for index in range(len(state)):
    phase = 1
    perm = mps.index_2_perm(index)

    # calculte the position of each 2 electrons a>b in complex plane and the phase between them
    for a in range(1 , len(perm)):
        for b in range(a):
            za = cite_index_2_z(perm[a], mps)
            zb = cite_index_2_z(perm[b], mps)
            phase *= np.exp(2j * np.angle(za - zb))
    state[index] *= phase

# %%
print("Hello world")
# %%
index = mps.perm_2_index((0,2,3,12))
number = index_2_state_vector_index(index, mps)

binary_string = bin(number)[2:].zfill(8)
print(binary_string)  # Output: '00001010'
np.array(binary_string)
indices = [i for i, bit in enumerate(binary_string[::-1]) if bit == '1']
print(indices)
#%%
indices = np.array(range(len(state)))
indices = indices[np.abs(state) > 1e-6]
# print(indices)

list = []
for i in indices:
    list.append(index_2_state_vector_index(i,mps))

print(np.array(sorted(list)))
indices = np.array(range(len(full_state_vector)))
indices = indices[np.abs(full_state_vector) > 1e-6]
print(indices - np.array(sorted(list)))
