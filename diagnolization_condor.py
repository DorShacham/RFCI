import os
import platform
import multiprocessing
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description='For a given Nx,Ny load the matrix if exisit and diagnolizing it')
    parser.add_argument('-Nx', type=int, help='Nx dimension of the lattice')
    parser.add_argument('-Ny', type=int, help='Ny dimension of the lattice')
    parser.add_argument('-cpu', type=int, help='number of cpus for multiprocess computation, if missing computes without multiprocess')
    parser.add_argument('--matrix_index', type=int, help='Index of the matrix in the specral flow matrices')

    args = parser.parse_args()
    Nx = args.Nx
    Ny = args.Ny
    n = Nx * Ny // 3
    cpu = args.cpu
    matrix_index = args.matrix_index

        # Check the operating system
    if platform.system() == "Linux":
        # Set environment variables to limit CPU usage on Linux
        os.environ["OMP_NUM_THREADS"] = str(cpu)
        os.environ["MKL_NUM_THREADS"] = str(cpu)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu)
        os.environ["JAX_NUM_THREADS"] = str(cpu)
        print(f"CPU usage limited to {cpu} threads on Linux.")
    elif platform.system() == "Darwin":
        # macOS-specific behavior (no limitation)
        print("Running on macOS. No CPU limitation applied.")
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())
    else:
        print("Operating system not recognized. No changes applied.")



    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import block_diag, dft, expm
    from tqdm import tqdm
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    from exact_diagnolization import *
    from IQH_state import *


    print('Loading matrix')
    H = sparse.load_npz(str(f'data/matrix/pectral_flow/H_Nx-{Nx}_Ny-{Ny}_{matrix_index}.npz'))
    interaction = sparse.load_npz(str(f'data/matrix/interactions_Nx-{Nx}_Ny-{Ny}.npz'))

    band_energy = 1
    interaction_strength = 2
    H = band_energy * (H + n * sparse.identity(n = np.shape(H)[0], format='csr'))  + interaction_strength * interaction
    #%%
    print('Diaganolizing')
    eigenvalues, eigenvectors = eigsh(H, k=4, which='SA')
    np.savez(f'data/states/spectral_flow/Nx-{Nx}_Ny-{Ny}_k-4.npz', a=eigenvectors, eigenvalues = eigenvalues)