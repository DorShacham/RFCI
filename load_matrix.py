#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
from tqdm import tqdm
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from IQH_state import *
from flux_attch import *

# Lattice shape
Nx = 3
Ny = 6
N = 2 * Nx * Ny

# Number of electrons
n = N//6
# n = 2
interaction_strength = 1e0
band_energy = 1e2
H_sb = build_H(Nx = Nx, Ny = Ny, band_energy = band_energy)

NN = []
for x in range(Nx):
    for y in range(Ny):
        n1 = cite_2_cite_index(x = x, y = y, sublattice = 0, Ny = Ny)
        for i in [0,1]:
            for j in [0,1]:
                n2 = cite_2_cite_index(x = (x - i) % Nx, y = (y - j) % Ny, sublattice = 1, Ny = Ny)
                NN.append((n1,n2))


mps = Multi_particle_state(N = N, n = n)
v = mps.zero_vector()

# From a list of (row, column, value) tuples
data_dict = {}


for index in tqdm(range(len(v))):
    state_perm = mps.index_2_perm(index)
    #signle_body_states is the indices of the sites where the electrons seats.
    for i in range(N):
        for j in range(N):
            if (i == j):
                if (index, index) in data_dict:
                    data_dict[(index,index)] += H_sb[i,j]  
                else:
                    data_dict[(index,index)] = H_sb[i,j]  
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
                    new_index = mps.perm_2_index(sorted_perm)
        
                    # Summing the new multi-particle state with the right coeff
                    if (new_index, index) in data_dict:
                        data_dict[(new_index,index)] += H_sb[i,j] * (-1)**k * (-1)**parity 
                    else:
                        data_dict[(new_index,index)] = H_sb[i,j] * (-1)**k * (-1)**parity 
    #interactions 
    for i,j in NN:
        if (i in state_perm) and  (j in state_perm):
            if (index, index) in data_dict:
                data_dict[(index,index)] += interaction_strength
            else:
                data_dict[(index,index)] = interaction_strength

rows, cols = zip(*data_dict.keys())
values = list(data_dict.values())
sparse_matrix = sparse.csc_matrix((values, (rows, cols)))



# Compute k largest eigenvalues and corresponding eigenvectors
k = 10  # Number of eigenvalues/vectors to compute
eigenvalues, eigenvectors = eigsh(sparse_matrix, k=k, which='SA')
print(sorted(eigenvalues))


eigenvalues = np.array(eigenvalues) 
# eigenvalues = eigenvalues - np.min(eigenvalues)
print(eigenvalues)

plt.figure()
plt.plot(np.ones(len(eigenvalues[:4])),eigenvalues[:4],".")

print_mp_state(eigenvectors[:,0],Nx,Ny,mps)
print_mp_state(eigenvectors[:,1],Nx,Ny,mps)
print_mp_state(eigenvectors[:,2],Nx,Ny,mps)
print_mp_state(eigenvectors[:,3],Nx,Ny,mps)

#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
from tqdm import tqdm
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import pickle

from IQH_state import *
from flux_attch import *

Nx = 3
Ny = 3
N = 2 * Nx * Ny

# Number of electrons
n = N // 6
mps = Multi_particle_state(N=N, n=n)

with open(f'files/data_dict_Nx-{Nx}_Ny-{Ny}.pickle', 'rb') as handle:
    loaded_dict = pickle.load(handle)

# sparse_matrix = sparse.load_npz(f'files/sparse_matrix_Nx-{Nx}_Ny-{Ny}.npz')
rows, cols = zip(*loaded_dict.keys())
values = list(loaded_dict.values())
sparse_matrix = sparse.csr_matrix((values, (rows, cols)))

# Compute k largest eigenvalues and corresponding eigenvectors
k = 10  # Number of eigenvalues/vectors to compute
eigenvalues, eigenvectors = eigsh(sparse_matrix, k=k, which='SA')
print(sorted(eigenvalues))


eigenvalues = np.array(eigenvalues) 
# eigenvalues = eigenvalues - np.min(eigenvalues)
print(eigenvalues)

plt.figure()
plt.plot(np.ones(len(eigenvalues[:4])),eigenvalues[:4],".")

print_mp_state(eigenvectors[:,0],Nx,Ny,mps)
print_mp_state(eigenvectors[:,1],Nx,Ny,mps)
print_mp_state(eigenvectors[:,2],Nx,Ny,mps)
print_mp_state(eigenvectors[:,3],Nx,Ny,mps)