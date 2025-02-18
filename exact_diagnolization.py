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
from multiprocessing import Manager, shared_memory
import os
import gc
from collections import ChainMap




from IQH_state import *
from flux_attch import *

def process_index(index,mps, H_sb, NN, interaction_strength, N, build = "interacting H"):
    try:
        hopping_terms = False
        interacting_terms = False
        local_pot = False

        if build == "interacting H":
            hopping_terms = True
            interacting_terms = True
        elif build == "interaction":
            interacting_terms = True
        elif build == "non interacting H":
            hopping_terms = True
        elif build == "local potential":
            local_pot = True
            pot_cite = 0

        state_perm = mps.index_2_perm(index)
        vector_size = mps.len
        sparse_col = sparse.dok_matrix((vector_size,1), dtype=np.complex128)
        # Single-body terms
        if hopping_terms:
            for i in range(N):
                for j in range(N):
                    if (i == j) and (i in state_perm):
                        sparse_col[index,0] += H_sb[i,j]
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
                            sparse_col[new_index,0] += H_sb[i,j] * (-1)**k * (-1)**parity 
            
        # Interaction terms
        if interacting_terms:
            for i,j in NN:
                if (i in state_perm) and (j in state_perm):
                    sparse_col[index,0] += interaction_strength

        # local potential terms
        if local_pot:
            if pot_cite in state_perm:
                sparse_col[index,0] += 1
        
        return (sparse_col.tocoo(), index)
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
        sparse_matrix = _build(Nx = Nx, Ny = Ny, n = n, H_sb = H_sb, band_energy= band_energy, phi = phi, phase_shift_x= phase_shift_x, phase_shift_y = phase_shift_y, element_cutoff = element_cutoff, multi_process = multi_process, max_workers = max_workers, multiprocess_func = multiprocess_func)

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


    if multi_process:
        if multiprocess_func is None:
            multiprocess_func = multiprocess_map

        manager = Manager()    
        NN_shared = manager.list(NN)  # Share NN as a list
        H_sb_np = np.array(H_sb)
        shm = shared_memory.SharedMemory(create=True, size=H_sb_np.nbytes)
        H_sb_shared = np.ndarray(H_sb_np.shape, dtype=H_sb_np.dtype, buffer=shm.buf)
        np.copyto(H_sb_shared, H_sb_np)

        # Prepare partial function with fixed arguments
        # Prepare partial function with read-only references
        process_index_partial = partial(
        process_index,
        mps=mps,
        H_sb=H_sb_shared,
        NN=NN_shared,
        interaction_strength=1,
        N=N,
        build= build
        )
        chunk_size = min(mps.len // max_workers, int(1e4))
        results = multiprocess_func(process_index_partial, range(mps.len), max_workers, chunk_size)
        
        manager.shutdown()
        shm.close()
        shm.unlink()
    
    else:
        process_index_partial = partial(
        process_index,
        mps=mps,
        H_sb=H_sb,
        NN=NN,
        interaction_strength=1,
        N=N,
        build= build
        )
        results = [process_index_partial(index) for index in tqdm(range(mps.len))]


    # Sort by index to ensure correct column order
    results.sort(key=lambda x: x[1])

    # Extract sorted columns and convert them to CSR format
    sorted_columns = [col for col, _ in results]

    

    # Horizontally stack columns to form the final CSR matrix
    sparse_matrix = sparse.hstack(sorted_columns, format='coo')
    sparse_matrix = sparse_matrix.tocsr()

    return sparse_matrix

def build_non_interacting_H(Nx, Ny, n = None, H_sb = None, band_energy = 1, phi =  np.pi/4, phase_shift_x = 0, phase_shift_y = 0, element_cutoff = None, multi_process = False, max_workers = 6, multiprocess_func=None):
    return _build(Nx = Nx, Ny = Ny, n = n, H_sb = H_sb, band_energy = band_energy, phi = phi, phase_shift_x = phase_shift_x, phase_shift_y = phase_shift_y, element_cutoff = element_cutoff, multi_process = multi_process, max_workers = max_workers, multiprocess_func = multiprocess_func, build = "non interacting H")

def build_interaction(Nx, Ny, n = None, H_sb = None, band_energy = 1, phi =  np.pi/4, phase_shift_x = 0, phase_shift_y = 0, element_cutoff = None, multi_process = False, max_workers = 6, multiprocess_func=None):
    return _build(Nx = Nx, Ny = Ny, n = n, H_sb = H_sb, band_energy = band_energy, phi = phi, phase_shift_x = phase_shift_x, phase_shift_y = phase_shift_y, element_cutoff = element_cutoff, multi_process = multi_process, max_workers = max_workers, multiprocess_func = multiprocess_func, build = "interaction")

def build_local_potential(Nx, Ny, n = None, H_sb = None, band_energy = 1, phi =  np.pi/4, phase_shift_x = 0, phase_shift_y = 0, element_cutoff = None, multi_process = False, max_workers = 6, multiprocess_func=None):
    return _build(Nx = Nx, Ny = Ny, n = n, H_sb = H_sb, band_energy = band_energy, phi = phi, phase_shift_x = phase_shift_x, phase_shift_y = phase_shift_y, element_cutoff = element_cutoff, multi_process = multi_process, max_workers = max_workers, multiprocess_func = multiprocess_func, build = "local potential")



if __name__ == "__main__":
    eigenvalues, eigenvectors = exact_diagnolization(Nx=2, Ny=6,interaction_strength=0.1,k=4, multi_process=True, max_workers=10, multiprocess_func=multiprocess_map,from_memory=False,save_result=False,show_result=True)
    print(eigenvalues)
