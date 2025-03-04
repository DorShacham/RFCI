import os
import platform
import multiprocessing
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description='For a given Nx,Ny load the matrix if exisit and diagnolizing it')
    parser.add_argument('-Nx', type=int, help='Nx dimension of the lattice')
    parser.add_argument('-Ny', type=int, help='Ny dimension of the lattice')
    parser.add_argument('-cpu', type=int, help='number of cpus for multiprocess computation, if missing computes without multiprocess')

    args = parser.parse_args()
    Nx = args.Nx
    Ny = args.Ny
    n = Nx * Ny // 3
    cpu = args.cpu

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

    phi_list = np.array(range(0,72 + 1)) / 72 * 3

    eigenvalues_list = []
    for i in tqdm(range(0,72 + 1)):
        loaded = np.load(f'/storage/ph_lindner/dorsh/RFCI/data/states/spectral_flow/Nx-{Nx}_Ny-{Ny}_{i}.npz')
        eigenvalues = loaded['eigenvalues']  
        eigenvalues_list.append(np.sort(eigenvalues))

    eigenvalues_list = np.array(eigenvalues_list) 
    plt.figure()
    plt.plot(phi_list,eigenvalues_list[:,0], "-.")
    plt.plot(phi_list,eigenvalues_list[:,1], "-.")
    plt.plot(phi_list,eigenvalues_list[:,2], "-.")
    plt.plot(phi_list,eigenvalues_list[:,3], "-.")
    plt.plot(phi_list,eigenvalues_list[:,4], "-.")
    plt.plot(phi_list,eigenvalues_list[:,5], "-.")
    plt.plot(phi_list,eigenvalues_list[:,6], "-.")
    plt.grid()
    
    plt.title(f"Spectral flow for {Nx,Ny} lattice \n(first 7 eigenvalues)")
    plt.savefig(f"./results/spectral_flow/Nx-{Nx}_Ny-{Ny}_k=7.jpg")


    plt.figure()
    plt.plot(phi_list,eigenvalues_list[:,0], "-.")
    plt.plot(phi_list,eigenvalues_list[:,1], "-.")
    plt.plot(phi_list,eigenvalues_list[:,2], "-.")
    plt.grid()

    plt.title(f"Spectral flow for {Nx,Ny} lattice \n(first 3 eigenvalues)")
    plt.savefig(f"./results/spectral_flow//Nx-{Nx}_Ny-{Ny}_k=3.jpg")
