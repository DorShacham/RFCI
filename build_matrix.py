import os
import platform
import multiprocessing


# Check the operating system
if platform.system() == "Linux":
    # Set environment variables to limit CPU usage on Linux
    os.environ["OMP_NUM_THREADS"] = "N"
    os.environ["MKL_NUM_THREADS"] = "N"
    os.environ["NUMEXPR_NUM_THREADS"] = "N"
    os.environ["OPENBLAS_NUM_THREADS"] = "N"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "N"
    os.environ["JAX_NUM_THREADS"] = "N"
    print("CPU usage limited to N threads on Linux.")
elif platform.system() == "Darwin":
    # macOS-specific behavior (no limitation)
    print("Running on macOS. No CPU limitation applied.")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())
else:
    print("Operating system not recognized. No changes applied.")


from argparse import ArgumentParser
import yaml
from exact_diagnolization import *

if __name__ == "__main__":
    parser = ArgumentParser(description='For a given Nx,Ny,n build the many body H and the interaction matrix interaction_strength = 1')
    parser.add_argument('--save_path', type=str, help='Path to the save location if is missing save at a deafult location')
    parser.add_argument('-Nx', type=int, help='Nx dimension of the lattice')
    parser.add_argument('-Ny', type=int, help='Ny dimension of the lattice')
    parser.add_argument('-n', type=int, help='number of electrons, if missing will taken to be Nx * Ny /3')
    parser.add_argument('-cpu', type=int, help='number of cpus for multiprocess computation, if missing computes without multiprocess')
    
    args = parser.parse_args()
    
    save_path = args.save_path
    Nx = args.Nx
    Ny = args.Ny
    n = args.n
    cpu = args.cpu

    if save_path is None:
        save_path = './data/matrix'

    if not (cpu is None):
        multi_process=True
        max_workers=cpu
        multiprocess_func=multiprocess_map
    else:
        multi_process=None
        max_workers=1
        multiprocess_func=None


    print("Building H")
    non_interacting_H = build_non_interacting_H(Nx, Ny, n=n, multi_process=multi_process,max_workers=max_workers,multiprocess_func=multiprocess_func)
    os.makedirs(save_path, exist_ok=True)
    print("Saving H")
    sparse.save_npz(save_path + str(f'/H_Nx-{Nx}_Ny-{Ny}.npz'), non_interacting_H)

    print("Building interactions")
    interaction = build_interaction(Nx, Ny, n=n, multi_process=multi_process,max_workers=max_workers,multiprocess_func=multiprocess_func)
    os.makedirs(save_path, exist_ok=True)
    print("Saving interactions")
    sparse.save_npz(save_path + str(f'/interactions_Nx-{Nx}_Ny-{Ny}.npz'), interaction)
