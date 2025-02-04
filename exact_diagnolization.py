#%%
# import os
# import platform
# import multiprocessing


# # Check the operating system
# if platform.system() == "Linux":
#     # Set environment variables to limit CPU usage on Linux
#     os.environ["OMP_NUM_THREADS"] = "N"
#     os.environ["MKL_NUM_THREADS"] = "N"
#     os.environ["NUMEXPR_NUM_THREADS"] = "N"
#     os.environ["OPENBLAS_NUM_THREADS"] = "N"
#     os.environ["VECLIB_MAXIMUM_THREADS"] = "N"
#     os.environ["JAX_NUM_THREADS"] = "N"
#     print("CPU usage limited to N threads on Linux.")
# elif platform.system() == "Darwin":
#     # macOS-specific behavior (no limitation)
#     print("Running on macOS. No CPU limitation applied.")
#     os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())
# else:
#     print("Operating system not recognized. No changes applied.")



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
import gc
from collections import ChainMap




from IQH_state import *
from flux_attch import *

def process_index(index,mps, H_sb, NN, interaction_strength, N, build = "interacting H"):
    try:
        if build == "interacting H":
            hopping_terms = True
            interacting_terms = True
        elif build == "interaction":
            hopping_terms = False
            interacting_terms = True
        elif build == "non interacting H":
            hopping_terms = True
            interacting_terms = False

        local_data_dict = {}
        state_perm = mps.index_2_perm(index)
        
        # Single-body terms
        if hopping_terms:
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
        if interacting_terms:
            for i,j in NN:
                if (i in state_perm) and (j in state_perm):
                    local_data_dict[(index, index)] = local_data_dict.get((index, index), 0) + interaction_strength
        
        return local_data_dict
    except Exception as e:
        print(f"Error in task: {e}")
        return None


def multiprocess_map(func, iterable, max_workers, chunk_size):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(tqdm(executor.map(func, iterable, chunksize=chunk_size), total=len(iterable)))


# compute the exact diagonalization of the problem with lattice of size @Nx,@Ny with @n electorn, if not spesficed then n = N //6 (1/3 filling of the lower band),
# @k - Number of eigenvalues/vectors to compute. @max_workers - If in multi-proccess mode, max number of workders.
# If @from_memory True, load sparse matrix from meomory and diagnolize it.
# return the eigenvalues, eigenvectors and save the results.
def exact_diagnolization(Nx, Ny, n = None, H_sb = None, band_energy = 1, interaction_strength = 1e-1, phi =  np.pi/4, phase_shift_x = 0, phase_shift_y = 0, element_cutoff = None  ,k = 10, multi_process = False, max_workers = 6, multiprocess_func=None, from_memory = False, save_result = True, show_result = True):

    path = str(f'results/Exact_Diagnolization/Nx-{Nx}_Ny-{Ny}')
    N = 2 * Nx * Ny
    if n is None:
        n = N // 6
    mps = Multi_particle_state(N=N, n=n)
    if not from_memory:
        # Number of electrons
        
        
        if H_sb is None:
            H_sb = build_H(Nx=Nx, Ny=Ny, band_energy = band_energy, phi=phi, phase_shift_x = phase_shift_x, phase_shift_y = phase_shift_y, element_cutoff=element_cutoff)

        NN = []
        for x in range(Nx):
            for y in range(Ny):
                n1 = cite_2_cite_index(x=x, y=y, sublattice=0, Ny=Ny)
                for delta_x,delta_y in [(0,0), (0,1), (-1,0), (-1,1)]:
                    n2 = cite_2_cite_index(x=(x + delta_x) % Nx, y=(y + delta_y) % Ny, sublattice=1, Ny=Ny)
                    NN.append((n1,n2))

        
        v = mps.zero_vector()

        # Prepare partial function with fixed arguments
        process_index_partial = partial(process_index, mps=mps, H_sb=H_sb, NN=NN, interaction_strength=interaction_strength, N=N)

        if multi_process:
            if multiprocess_func is None:
                multiprocess_func = multiprocess_map
            chunk_size = min(len(v) // max_workers, int(1e4))
            results = multiprocess_func(process_index_partial, range(len(v)), max_workers, chunk_size)
        else:
            results = [process_index_partial(index) for index in tqdm(range(len(v)))]

        # Combine results
        # data_dict = dict(ChainMap(*results))
        data_dict =  {k: v for d in results for k, v in d.items()}
        rows, cols = zip(*data_dict.keys())
        values = list(data_dict.values())
        sparse_matrix = sparse.csr_matrix((values, (rows, cols)))

        if save_result:
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

    if show_result or save_result:
        plt.figure()
        plt.plot(np.ones(len(eigenvalues)), eigenvalues, ".")
    if save_result:
        plt.savefig(path + str('/eigenvalues.jpg'))
        print_mp_state(eigenvectors[:,0],Nx,Ny,mps,saveto= path + str("/ev0.jpg"))
        print_mp_state(eigenvectors[:,1],Nx,Ny,mps,saveto= path + str("/ev1.jpg"))
        print_mp_state(eigenvectors[:,2],Nx,Ny,mps,saveto= path + str("/ev2.jpg"))
        print_mp_state(eigenvectors[:,3],Nx,Ny,mps,saveto= path + str("/ev3.jpg"))
        
        with open(path + str('/data.txt'), 'w') as file:
            file_dict = {"Nx":Nx, "Ny":Ny, "n":n, "band_energy": band_energy, "interaction_strength":interaction_strength,"eigenvalues":eigenvalues}
            file.write(str(file_dict))
    elif show_result:
        print_mp_state(eigenvectors[:,0],Nx,Ny,mps,saveto= None)
        print_mp_state(eigenvectors[:,1],Nx,Ny,mps,saveto= None)
        print_mp_state(eigenvectors[:,2],Nx,Ny,mps,saveto= None)
        print_mp_state(eigenvectors[:,3],Nx,Ny,mps,saveto= None)

    

    return eigenvalues, eigenvectors

# builds many body H or many without interaction or only interaction term with interaction_strength = 1
def _build(Nx, Ny, n = None, H_sb = None, band_energy = 1, phi =  np.pi/4, phase_shift_x = 0, phase_shift_y = 0, element_cutoff = None, multi_process = False, max_workers = 6, multiprocess_func=None, build = "interacting H"):
    N = 2 * Nx * Ny
    if n is None:
        n = N // 6
    mps = Multi_particle_state(N=N, n=n)
    # Number of electrons
        
        
    if H_sb is None:
        H_sb = build_H(Nx=Nx, Ny=Ny, band_energy = band_energy, phi=phi, phase_shift_x = phase_shift_x, phase_shift_y = phase_shift_y, element_cutoff=element_cutoff)

    NN = []
    for x in range(Nx):
        for y in range(Ny):
            n1 = cite_2_cite_index(x=x, y=y, sublattice=0, Ny=Ny)
            for delta_x,delta_y in [(0,0), (0,1), (-1,0), (-1,1)]:
                n2 = cite_2_cite_index(x=(x + delta_x) % Nx, y=(y + delta_y) % Ny, sublattice=1, Ny=Ny)
                NN.append((n1,n2))

    
    v = mps.zero_vector()

    # Prepare partial function with fixed arguments
    process_index_partial = partial(process_index, mps=mps, H_sb=H_sb, NN=NN, interaction_strength=1, N=N, build = build)

    if multi_process:
        if multiprocess_func is None:
            multiprocess_func = multiprocess_map
        chunk_size = min(len(v) // max_workers, int(1e4))
        results = multiprocess_func(process_index_partial, range(len(v)), max_workers, chunk_size)
    else:
        results = [process_index_partial(index) for index in tqdm(range(len(v)))]

    # Combine results
    # data_dict = dict(ChainMap(*results))
    data_dict =  {k: v for d in results for k, v in d.items()}
    rows, cols = zip(*data_dict.keys())
    values = list(data_dict.values())
    sparse_matrix = sparse.csr_matrix((values, (rows, cols)))

    return sparse_matrix

def build_non_interacting_H(Nx, Ny, n = None, H_sb = None, band_energy = 1, phi =  np.pi/4, phase_shift_x = 0, phase_shift_y = 0, element_cutoff = None, multi_process = False, max_workers = 6, multiprocess_func=None):
    return _build(Nx, Ny, n, H_sb, band_energy, phi, phase_shift_x, phase_shift_y, element_cutoff, multi_process, max_workers, multiprocess_func, build = "non interacting H")

def build_interaction(Nx, Ny, n = None, H_sb = None, band_energy = 1, phi =  np.pi/4, phase_shift_x = 0, phase_shift_y = 0, element_cutoff = None, multi_process = False, max_workers = 6, multiprocess_func=None):
    return _build(Nx, Ny, n, H_sb, band_energy, phi, phase_shift_x, phase_shift_y, element_cutoff, multi_process, max_workers, multiprocess_func, build = "interaction")



if __name__ == "__main__":
    eigenvalues, eigenvectors = exact_diagnolization(Nx=2, Ny=6,k=4, multi_process=False, max_workers=10, multiprocess_func=multiprocess_map,from_memory=False,save_result=True,show_result=False)