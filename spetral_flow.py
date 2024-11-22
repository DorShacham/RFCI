#%%
from IQH_state import *
from flux_attch import *
from exact_diagnolization import *
import numpy as np

Nx = 3
Ny = 3

for cutoff in [1e-1, 1e-2, 1e-3, None]:
    element_cutoff = cutoff

    phi_list = np.linspace(start=0,stop=3, num=30 + 1)
    eigenvalues_list = []
    for i, phi in enumerate(phi_list):
        eigenvalues, eigenvectors = exact_diagnolization(Nx=Nx, Ny=Ny,phase_shift_x=phi * 2 * np.pi, element_cutoff=element_cutoff ,k=7, multi_process=False,multiprocess_func=multiprocess_map, save_result= False, show_result=False)
        eigenvalues_list.append(eigenvalues)
        print(i)

    eigenvalues_list = np.array(eigenvalues_list) 
    eigenvalues_list = eigenvalues_list - np.min(eigenvalues_list)
    plt.figure()
    plt.plot(phi_list,eigenvalues_list[:,0], "-.")
    plt.plot(phi_list,eigenvalues_list[:,1], "-.")
    plt.plot(phi_list,eigenvalues_list[:,2], "-.")
    plt.plot(phi_list,eigenvalues_list[:,3], "-.")
    plt.plot(phi_list,eigenvalues_list[:,4], "-.")
    plt.plot(phi_list,eigenvalues_list[:,5], "-.")
    plt.plot(phi_list,eigenvalues_list[:,6], "-.")
    plt.grid()
    
    plt.title(f"Spectral flow with element cutoff\n of {cutoff} for ({Nx,Ny}) lattice \n(first 7 eigenvalues shifted by the lowest value)")
    plt.savefig(f"./results/spectral_flow/Nx-{Nx}_Ny-{Ny}/H_element_cutoff-{cutoff}_1.jpg")



    plt.figure()
    plt.plot(phi_list,eigenvalues_list[:,0], "-.")
    plt.plot(phi_list,eigenvalues_list[:,1], "-.")
    plt.plot(phi_list,eigenvalues_list[:,2], "-.")
    plt.grid()

    plt.title(f"Spectral flow with element cutoff\n of {cutoff} for ({Nx,Ny}) lattice \n(first 3 eigenvalues shifted by the lowest value)")
    plt.savefig(f"./results/spectral_flow/Nx-{Nx}_Ny-{Ny}/H_element_cutoff-{cutoff}_2.jpg")
