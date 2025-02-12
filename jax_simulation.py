#%%
from scipy.linalg import expm
import os
import jax
import jax.numpy as np
from  jax.experimental import sparse as jax_sparse
from  scipy import sparse 
from mpmath import *


from IQH_state import *
from flux_attch import *
from jax_vqe import *
from exact_diagnolization import *
from jax_ansatz import Jax_ansatz
from chern_ins_magnetic import add_magnetic_field



def build_H_many_body(Nx, Ny, interaction_strength = 1e-1, band_energy = 1):
    H_many_body = sparse.load_npz(f'results/Exact_Diagnolization/Nx-{Nx}_Ny-{Ny}/sparse_matrix.npz')
    return H_many_body



# for a given @state_vecotr on lattice @Nx,@Ny print a heatmap of the distribution of electrons.
# if @saveto is not None should be a path to save location for the heatmap

def vqe_simulation(Nx, Ny, config_list, n = None, p=-1, q=3 , pre_ansatz = None,saveto = None, log = False):
    # Initialzing state
    if n is None:
        n = Nx * Ny // q

    try:
        IQH_state = np.load(f'data/states/Nx-{Nx}_Ny-{Ny}_q={q}_magnetic.npy')
        mps = Multi_particle_state(N = 2 * Nx * Ny,n=n)
    except:
        print("Calculting IQH state")
        H_real_space = build_H(Nx,Ny)
        H_real_space_magnetic = add_magnetic_field(np.array(H_real_space), p, q, Nx, Ny, cites_per_uc = 2)
        IQH_state, mps = create_IQH_in_extendend_lattice(Nx,Ny,n,extention_factor = 1, band_energy = 1, H_sb = H_real_space_magnetic)
        np.save(f'data/states/Nx-{Nx}_Ny-{Ny}_q={q}_magnetic.npy',IQH_state)

    state = IQH_state

    try:
        H = sparse.load_npz(str(f'data/matrix/H_Nx-{Nx}_Ny-{Ny}.npz'))
        interaction = sparse.load_npz(str(f'data/matrix/interactions_Nx-{Nx}_Ny-{Ny}.npz'))
        H_many_body = jax_sparse.BCOO.from_scipy_sparse(H + config_dict['interaction_strength'] * interaction)
    except:
        print("Calculting H_many_body matrix")
        H = build_non_interacting_H(Nx = Nx, Ny = Ny, n = n, multi_process= False)
        sparse.save_npz(str(f'data/matrix/H_Nx-{Nx}_Ny-{Ny}.npz'), H)

        interaction = build_interaction(Nx = Nx, Ny = Ny, n = n, multi_process= False)
        sparse.save_npz(str(f'data/matrix/interactions_Nx-{Nx}_Ny-{Ny}.npz'), interaction)
        
        H_many_body = H + config_dict['interaction_strength'] * interaction
        eigenvalues, eigenvectors = eigenvalues, eigenvectors = eigsh(H_many_body, k=4, which='SA')
        np.savez(f'data/states/Nx-{Nx}_Ny-{Ny}_k-4.npz', a=eigenvectors)

        H_many_body = jax_sparse.BCOO.from_scipy_sparse(H_many_body)

    

    for i, config_dict in enumerate(config_list):
        config_dict['Nx'] = Nx
        config_dict['Ny'] = Ny
        config_dict['mps'] = mps
        config_dict['config_i'] = i
        config_dict['log'] = log
        config_dict['hamiltonian'] = H_many_body
        config_dict['initial_state'] = state
        
        if saveto is not None:
            path = str(saveto) + str(f'/optimization_{i}')
            os.makedirs(path, exist_ok=True)
            config_dict['saveto'] = path
        else:
            path = None

        if config_dict['ground_state_degeneracy'] is not None:
            ground_state_degeneracy = config_dict['ground_state_degeneracy']
            try:
                loaded = np.load(f'data/states/Nx-{Nx}_Ny-{Ny}_k-4.npz')
                eigenvectors = loaded['a']            
            except:
                eigenvalues, eigenvectors = exact_diagnolization(Nx, Ny, band_energy=config_dict['band_energy'], interaction_strength=config_dict['interaction_strength'],k=config_dict['ground_state_degeneracy'],multi_process=False, save_result=False, show_result=False)
            eigenvectors = eigenvectors[:,:ground_state_degeneracy].T
        else:
            eigenvectors = None
            
        config_dict['ground_states'] = eigenvectors


        ansatz = Jax_ansatz(Nx = Nx, Ny = Ny, n=n, local_layers_num=config_dict['layer_numer'])
        config_dict['ansatz'] = ansatz
        
        vqe = VQE(config_dict)
        res = vqe.minimize()
        vqe.plot()
        # calculting initial and final energy
        i_state = state
        f_state = ansatz.apply_ansatz(params = res.x, state = state)

        initial_energy = my_estimator(i_state,H_many_body)
        finial_energy = my_estimator(f_state,H_many_body)
        if saveto is not None:
            print_mp_state(i_state,Nx,Ny,mps,saveto=str(path) + str('/initial_state.jpg'))
            print_mp_state(f_state,Nx,Ny,mps,saveto=str(path) + str('/final_state.jpg'))
            if log:
                wandb.log({"Electron Density": wandb.Image(str(path) + str('/initial_state.jpg'), caption=f"Initial state config {i}")})
                wandb.log({"Electron Density": wandb.Image(str(path) + str('/final_state.jpg'), caption=f"Final state config {i}")})

            with open(path + str('/data.txt'), 'w') as file:
                file.write(str(config_dict))
                file.write(f"\ninitial_energy = {initial_energy.real}")
                file.write(f"\nfinial_energy = {finial_energy.real}")
                file.write(f"\noptimization solution = {res}")
        else:
            print_mp_state(i_state,Nx,Ny,mps)
            print_mp_state(f_state,Nx,Ny,mps)
            print(f"initial_energy = {initial_energy}")
            print(f"finial_energy = {finial_energy}")
            print(f"optimization solution = {res}")

        state = f_state
