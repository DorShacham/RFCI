#%% 
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
from tqdm import tqdm

from IQH_state import *


#%%


# given a state @state and its @mps of the compact Hilbert space (size of N Choose n) calculate the new state in that space with flux attached.
def flux_attch_2_compact_state(state, mps, Ny):
    for index in range(len(state)):
        phase = 1
        perm = mps.index_2_perm(index)

        # calculte the position of each 2 electrons a>b in complex plane and the phase between them
        for a in range(1 , len(perm)):
            for b in range(a):
                za = cite_index_2_z(perm[a], mps, Ny)
                zb = cite_index_2_z(perm[b], mps, Ny)
                phase *= np.exp(2j * np.angle(za - zb))
        state[index] *= phase
    return state


# #%%
# # Preparing the Full Hilbert 2^N state
# Nx = 2
# Ny = 2
# extention_factor = 3

# state, mps = create_IQH_in_extendend_lattice(Nx = Nx, Ny = Ny, extention_factor = extention_factor)

# Nx = extention_factor * Nx
# N = 2 * Nx * Ny

# #%%
# # Fulx attachment [without qiskit]

# # Calcute the accumlated phase for each cell
# for index in range(len(state)):
#     phase = 1
#     perm = mps.index_2_perm(index)

#     # calculte the position of each 2 electrons a>b in complex plane and the phase between them
#     for a in range(1 , len(perm)):
#         for b in range(a):
#             za = cite_index_2_z(perm[a], mps, Ny)
#             zb = cite_index_2_z(perm[b], mps, Ny)
#             phase *= np.exp(2j * np.angle(za - zb))
#     state[index] *= phase

# # %%
# print("Hello world")
# # %%
# index = mps.perm_2_index((0,2,3,12))
# number = index_2_state_vector_index(index, mps)

# binary_string = bin(number)[2:].zfill(8)
# print(binary_string)  # Output: '00001010'
# np.array(binary_string)
# indices = [i for i, bit in enumerate(binary_string[::-1]) if bit == '1']
# print(indices)
# #%%
# indices = np.array(range(len(state)))
# indices = indices[np.abs(state) > 1e-6]
# # print(indices)

# list = []
# for i in indices:
#     list.append(index_2_state_vector_index(i,mps))

# print(np.array(sorted(list)))
# indices = np.array(range(len(full_state_vector)))
# indices = indices[np.abs(full_state_vector) > 1e-6]
# print(indices - np.array(sorted(list)))
