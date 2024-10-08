#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag, dft, expm
from tqdm import tqdm
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import concurrent.futures
from functools import partial
import pickle


from IQH_state import *
from flux_attch import *

def process_index(index,mps, H_sb, NN, interaction_strength, N):
    try:
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
    except Exception as e:
        print(f"Error in task: {e}")
        return None



def combine_results(results):
    data_dict = {}
    for result in results:
        for key, value in result.items():
            if key in data_dict:
                data_dict[key] += value
            else:
                data_dict[key] = value
    return data_dict

# Main script
if __name__ == "__main__":
    # Latice shape
    Nx = 3
    Ny = 6
    N = 2 * Nx * Ny

    # Number of electrons
    n = N // 6
    band_energy = 1e2
    interaction_strength = 1e-1 

    H_sb = build_H(Nx=Nx, Ny=Ny, band_energy = 1e2)

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

    # Use ProcessPoolExecutor for multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        chunk_size = len(v) // 12  # Adjust chunk size based on your specific case
        results = list(tqdm(executor.map(process_index_partial, range(len(v)), chunksize=min(chunk_size,int(12e3))), total=len(v)))

    # Combine results
    data_dict = combine_results(results)
    with open(f'files/data_dict_Nx-{Nx}_Ny-{Ny}.pickle', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows, cols = zip(*data_dict.keys())
    values = list(data_dict.values())
    sparse_matrix = sparse.csr_matrix((values, (rows, cols)))
    # sparse.save_npz(f'files/sparse_matrix_Nx-{Nx}_Ny-{Ny}.npz', sparse_matrix)


    # Compute k largest eigenvalues and corresponding eigenvectors
    k = 10  # Number of eigenvalues/vectors to compute
    eigenvalues, eigenvectors = eigsh(sparse_matrix, k=k, which='SA')

    eigenvalues = np.array(eigenvalues) 
    print(sorted(eigenvalues))

    plt.figure()
    plt.plot(np.ones(len(eigenvalues)), eigenvalues, ".")
    plt.show()

    print_mp_state(eigenvectors[:,0],Nx,Ny,mps)
    print_mp_state(eigenvectors[:,1],Nx,Ny,mps)
    print_mp_state(eigenvectors[:,2],Nx,Ny,mps)
    print_mp_state(eigenvectors[:,3],Nx,Ny,mps)

#%%

