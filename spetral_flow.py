#%%
from IQH_state import *
from flux_attch import *
from exact_diagnolization import *
import numpy as np

Nx = 2
Ny = 6

for interaction_strength in [0]:

    phi_list = np.linspace(start=0,stop=3, num=12 + 1)
    eigenvalues_list = []
    for i, phi in enumerate(phi_list):
        eigenvalues, eigenvectors = exact_diagnolization(Nx=Nx, Ny=Ny,phase_shift_y=phi * 2 * np.pi, interaction_strength=interaction_strength ,k=7, multi_process=False,multiprocess_func=multiprocess_map, save_result= False, show_result=False)
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
    
    plt.title(f"Spectral flow with interaction strentgh\n of {interaction_strength} for ({Nx,Ny}) lattice \n(first 7 eigenvalues shifted by the lowest value)")
    plt.savefig(f"./results/spectral_flow/interaction_shift/Nx-{Nx}_Ny-{Ny}/1_interaction-{interaction_strength}.jpg")



    plt.figure()
    plt.plot(phi_list,eigenvalues_list[:,0], "-.")
    plt.plot(phi_list,eigenvalues_list[:,1], "-.")
    plt.plot(phi_list,eigenvalues_list[:,2], "-.")
    plt.grid()

    plt.title(f"Spectral flow with interaction strentgh\n of {interaction_strength} for ({Nx,Ny}) lattice \n(first 3 eigenvalues shifted by the lowest value)")
    plt.savefig(f"./results/spectral_flow/interaction_shift/Nx-{Nx}_Ny-{Ny}/2_interaction-{interaction_strength}.jpg")
