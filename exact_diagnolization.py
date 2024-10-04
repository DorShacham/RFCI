#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
from tqdm import tqdm
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import concurrent.futures
from functools import partial


from IQH_state import *
from flux_attch import *



def process_index(index, mps, H_sb, NN, interaction_strength, N):
    local_data_dict = {}
    state_perm = mps.index_2_perm(index)
    
    # Single-body terms
    for i in range(N):
        for j in range(N):
            if i == j:
                local_data_dict[(index, index)] = local_data_dict.get((index, index), 0) + H_sb[i,j]
            else:
                if (i in state_perm) or (not (j in state_perm)):
                    continue 
                else:
                    k = state_perm.index(j)
                    new_perm = list(state_perm)
                    del new_perm[k]
                    new_perm.insert(0,i)
                    parity, sorted_perm = permutation_parity(tuple(new_perm), return_sorted_array=True)
                    new_index = mps.perm_2_index(sorted_perm)
                    
                    local_data_dict[(new_index, index)] = local_data_dict.get((new_index, index), 0) + H_sb[i,j] * (-1)**k * (-1)**parity 
    
    # Interaction terms
    for i,j in NN:
        if (i in state_perm) and (j in state_perm):
            local_data_dict[(index, index)] = local_data_dict.get((index, index), 0) + interaction_strength
    
    return local_data_dict

# Main script
if __name__ == "__main__":
    # Latice shape
    Nx = 6
    Ny = 3
    N = 2 * Nx * Ny

    # Number of electrons
    n = N // 6
    interaction_strength = 1e-1

    H_sb = build_H(Nx=Ny, Ny=Nx)

    NN = []
    for x in range(Nx):
        for y in range(Ny):
            n1 = cite_2_cite_index(x=x, y=y, sublattice=0, Ny=Ny)
            for i in [0,1]:
                for j in [0,1]:
                    n2 = cite_2_cite_index(x=(x - i) % Nx, y=(y - j) % Ny, sublattice=1, Ny=Ny)
                    NN.append((n1,n2))

    mps = Multi_particle_state(N=N, n=n)
    v = mps.zero_vector()

    # Prepare partial function with fixed arguments
    process_index_partial = partial(process_index, mps=mps, H_sb=H_sb, NN=NN, interaction_strength=interaction_strength, N=N)

    # Use ThreadPoolExecutor for multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_index_partial, range(len(v))), total=len(v)))

    # Combine results
    data_dict = {}
    for result in results:
        for key, value in result.items():
            if key in data_dict:
                data_dict[key] += value
            else:
                data_dict[key] = value

    rows, cols = zip(*data_dict.keys())
    values = list(data_dict.values())
    sparse_matrix = sparse.csc_matrix((values, (rows, cols)))

    # Compute k largest eigenvalues and corresponding eigenvectors
    k = 10  # Number of eigenvalues/vectors to compute
    eigenvalues, eigenvectors = eigsh(sparse_matrix, k=k, which='SA')

    eigenvalues = np.array(eigenvalues) 
    print(eigenvalues)

    plt.figure()
    plt.plot(np.ones(len(eigenvalues)), eigenvalues, ".")
    plt.show()



#%%

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
Nx = 2
Ny = 3
N = 2 * Nx * Ny

# Number of electrons
n = N//6
# n = 2
interaction_strength = 1e-1

H_sb = build_H(Nx = Ny, Ny = Nx)

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