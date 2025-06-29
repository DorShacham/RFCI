import os
import platform
import multiprocessing



from argparse import ArgumentParser
import yaml
from exact_diagnolization import *

if __name__ == "__main__":
    parser = ArgumentParser(description='For a given Nx,Ny,n build the many body H and the interaction matrix interaction_strength = 1')
    parser.add_argument('--save_path', type=str, help='Path to the save location if is missing save at a deafult location')
    parser.add_argument('--matrix_type', type=str, help='"H" - for non interacting Hamiltonian, "inter" - for interactins, "loc" - for local potential')
    parser.add_argument('--sp', action='store_true', help='Calculate H with different phi y for spectal flow')
    parser.add_argument('--phi', type=float, help='Inserting flux in the y direction')
    parser.add_argument('-M', type=float, help='Add a mass term which can drive toplogical phase transiation')
    parser.add_argument('--name', type=str, help='Add an odditonal string to the file name')
    parser.add_argument('--naive', action='store_true', help='Creats real hopping for trival lattice')
    parser.add_argument('-Nx', type=int, help='Nx dimension of the lattice')
    parser.add_argument('-Ny', type=int, help='Ny dimension of the lattice')
    parser.add_argument('-n', type=int, help='number of electrons, if missing will taken to be Nx * Ny /3')
    parser.add_argument('-cpu', type=int, help='number of cpus for multiprocess computation, if missing computes without multiprocess')

    
    args = parser.parse_args()
    
    save_path = args.save_path
    matrix_type = args.matrix_type
    phi_y = args.phi
    Nx = args.Nx
    Ny = args.Ny
    n = args.n
    cpu = args.cpu
    spetral_flow = args.sp
    naive = args.naive

    if args.M is None:
        M = 0
    else:
        M = args.M
    


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
        cpu = 1

    build_H = False
    build_inter = False
    build_loc = False

    if matrix_type is None:
        build_H = True
        build_inter = True
    elif matrix_type == "H":
        build_H = True
    elif matrix_type == "inter":
        build_inter = True
    elif matrix_type == "loc":
        build_loc = True

    
    

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

    if spetral_flow:
        build_inter = True
        build_H = False
        save_path = './data/matrix/spectral_flow'
        for i,phi_y in enumerate(np.linspace(0,3,72 + 1)):
            non_interacting_H = build_non_interacting_H(Nx, Ny, n=n,phase_shift_y= phi_y * 2 * np.pi, multi_process=multi_process,max_workers=max_workers,multiprocess_func=multiprocess_func, M = M)
            os.makedirs(save_path, exist_ok=True)
            print(f"Saving H {i} / 73")
            sparse.save_npz(save_path + str(f'/H_Nx-{Nx}_Ny-{Ny}_{i}.npz'), non_interacting_H)
            

    if build_H:
        print("Building H")
        if not (phi_y is None):
            non_interacting_H = build_non_interacting_H(Nx, Ny, n=n,phase_shift_x=  0 ,phase_shift_y=  2 * np.pi * phi_y , multi_process=multi_process,max_workers=max_workers,multiprocess_func=multiprocess_func, M = M)
            # non_interacting_H = build_non_interacting_H(Nx, Ny, n=n,phase_shift_x=  2 * np.pi * phi_y ,phase_shift_y= 0 , multi_process=multi_process,max_workers=max_workers,multiprocess_func=multiprocess_func, M = M)
        else:
            non_interacting_H = build_non_interacting_H(Nx, Ny, n=n, multi_process=multi_process,max_workers=max_workers,multiprocess_func=multiprocess_func, M = M, naive=naive)

        print("Saving H")
        os.makedirs(save_path, exist_ok=True)
        if args.name is not None:
            sparse.save_npz(save_path + str(f'/H_Nx-{Nx}_Ny-{Ny}_{args.name}.npz'), non_interacting_H)
        else:
            sparse.save_npz(save_path + str(f'/H_Nx-{Nx}_Ny-{Ny}.npz'), non_interacting_H)


        
        

    if build_inter:
        print("Building interactions")
        interaction = build_interaction(Nx, Ny, n=n, multi_process=multi_process,max_workers=max_workers,multiprocess_func=multiprocess_func)
        os.makedirs(save_path, exist_ok=True)
        print("Saving interactions")
        sparse.save_npz(save_path + str(f'/interactions_Nx-{Nx}_Ny-{Ny}.npz'), interaction)

    if build_loc:
        print("Building local potential")
        pot = build_local_potential(Nx, Ny, n=n, multi_process=multi_process,max_workers=max_workers,multiprocess_func=multiprocess_func)
        os.makedirs(save_path, exist_ok=True)
        print("Saving local potential")
        sparse.save_npz(save_path + str(f'/local_potential_Nx-{Nx}_Ny-{Ny}.npz'), pot)
