
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
import os


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

def multiprocess_map(func, iterable, max_workers, chunk_size):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(tqdm(executor.map(func, iterable, chunksize=chunk_size), total=len(iterable)))


# compute the exact diagonalization of the problem with lattice of size @Nx,@Ny with @n electorn, if not spesficed then n = N //6 (1/3 filling of the lower band),
# @k - Number of eigenvalues/vectors to compute. @max_workers - If in multi-proccess mode, max number of workders.
# If @from_memory True, load sparse matrix from meomory and diagnolize it.
# return the eigenvalues, eigenvectors and save the results.
def exact_diagnolization(Nx, Ny, n = None, band_energy = 1, interaction_strength = 1e-1, k = 10, multi_process = True, max_workers = 4, multiprocess_func=None, from_memory = False):

    N = 2 * Nx * Ny
    path = str(f'results/Exact_Diagnolization/Nx-{Nx}_Ny-{Ny}')

    if not from_memory:
        # Number of electrons
        if n is None:
            n = N // 6
        

        H_sb = build_H(Nx=Nx, Ny=Ny, band_energy = band_energy)

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

        if multi_process:
            if multiprocess_func is None:
                multiprocess_func = multiprocess_map
            chunk_size = min(len(v) // max_workers, int(4e3))
            results = multiprocess_func(process_index_partial, range(len(v)), max_workers, chunk_size)
        else:
            results = [process_index_partial(index) for index in tqdm(range(len(v)))]

        # Combine results
        data_dict = combine_results(results)

        rows, cols = zip(*data_dict.keys())
        values = list(data_dict.values())
        sparse_matrix = sparse.csr_matrix((values, (rows, cols)))

        os.makedirs(path, exist_ok=True)
        sparse.save_npz(path + str('/sparse_matrix.npz'), sparse_matrix)

    else: # loading from memory
        sparse_matrix = sparse.load_npz(path + str('/sparse_matrix.npz'))


    eigenvalues, eigenvectors = eigsh(sparse_matrix, k=k, which='SA')
    
    eigen_pairs = list(zip(eigenvalues, eigenvectors.T))
    eigen_pairs.sort(key=lambda x: x[0])
    eigenvalues, eigenvectors = zip(*eigen_pairs)
    eigenvalues = np.array(eigenvalues)
    eigenvectors = np.array(eigenvectors).T  # Transpose back to original shape

    plt.figure()
    plt.plot(np.ones(len(eigenvalues)), eigenvalues, ".")
    plt.savefig(path + str('/eigenvalues.jpg'))

    print_mp_state(eigenvectors[:,0],Nx,Ny,mps,saveto= path + str("/ev0.jpg"))
    print_mp_state(eigenvectors[:,1],Nx,Ny,mps,saveto= path + str("/ev1.jpg"))
    print_mp_state(eigenvectors[:,2],Nx,Ny,mps,saveto= path + str("/ev2.jpg"))
    print_mp_state(eigenvectors[:,3],Nx,Ny,mps,saveto= path + str("/ev3.jpg"))

    with open(path + str('/data.txt'), 'w') as file:
        file_dict = {"Nx":Nx, "Ny":Ny, "n":n, "band_energy": band_energy, "interaction_strength":interaction_strength,"eigenvalues":eigenvalues}
        file.write(str(file_dict))

    return eigenvalues, eigenvectors


if __name__ == "__main__":
    exact_diagnolization()