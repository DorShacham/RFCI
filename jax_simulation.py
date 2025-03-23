#%%
from scipy.linalg import expm
import os
from jax import config
# Enable 64-bit computation
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from  jax.experimental import sparse as jax_sparse
from  scipy import sparse 
from mpmath import *


from IQH_state import *
from flux_attch import *
from jax_vqe import *
from exact_diagnolization import *
from jax_ansatz import Jax_ansatz
from chern_ins_magnetic import add_magnetic_field_chern



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
        IQH_state = jnp.load(f'data/states/Nx-{Nx}_Ny-{Ny}_q={q}_magnetic.npy')
        mps = Multi_particle_state(N = 2 * Nx * Ny,n=n)
    except:
        print("Calculting IQH state")
        H_real_space = build_H(Nx,Ny)
        H_real_space_magnetic = add_magnetic_field_chern(jnp.array(H_real_space), p, q, Nx, Ny)
        IQH_state, mps = create_IQH_in_extendend_lattice(Nx,Ny,n,extention_factor = 1, band_energy = 1, H_sb = H_real_space_magnetic)
        jnp.save(str(f'data/states/Nx-{Nx}_Ny-{Ny}_q={q}_magnetic.npy'),IQH_state)

    a = 2 # control the momentum sector on the symmetric state
    state = IQH_state
    T_y_expectation = state.T.conjugate() @ translation_operator(state,mps,Nx,Ny,Tx=0,Ty=1)
    if jnp.abs(jnp.abs(T_y_expectation) - 1) > 1e-5: # state is not symmetric
        # sym state = phase * (phase ** (-1/3) I + phase ** (-2/3) T + phase ** (-3/3) T^2) state
        phase = state.T.conjugate() @ translation_operator(state,mps,Nx,Ny,Tx=0,Ty=3)
        sym_state = jnp.array(state) * (phase ** (-1/3)) * jnp.exp(1j * 2 * jnp.pi / (-3) * a)
        for i in range(1,q):
            sym_state += (translation_operator(state,mps,Nx,Ny,Tx=0,Ty=i)) * (phase ** (-(i+1)/3)) * jnp.exp(1j * 2 * jnp.pi / (-3) * (i+1) * a)
        sym_state = normalize(sym_state * phase)
        state = sym_state

    try:
        H = sparse.load_npz(str(f'data/matrix/H_Nx-{Nx}_Ny-{Ny}.npz'))
        interaction = sparse.load_npz(str(f'data/matrix/interactions_Nx-{Nx}_Ny-{Ny}.npz'))
        H_many_body =  float(config_list[0]['band_energy']) * (H + (Nx * Ny // 3) * sparse.identity(n = jnp.shape(H)[0], format='csr'))  + config_list[0]['interaction_strength'] * interaction
    except:
        print("Calculting H_many_body matrix")
        H = build_non_interacting_H(Nx = Nx, Ny = Ny, n = n, multi_process= False)
        sparse.save_npz(str(f'data/matrix/H_Nx-{Nx}_Ny-{Ny}.npz'), H)

        interaction = build_interaction(Nx = Nx, Ny = Ny, n = n, multi_process= False)
        sparse.save_npz(str(f'data/matrix/interactions_Nx-{Nx}_Ny-{Ny}.npz'), interaction)
        
        H_many_body =  float(config_list[0]['band_energy']) * (H + n * sparse.identity(n = jnp.shape(H)[0], format='csr'))  + config_list[0]['interaction_strength'] * interaction
        eigenvalues, eigenvectors = eigenvalues, eigenvectors = eigsh(H_many_body, k=4, which='SA')
        jnp.savez(f'data/states/Nx-{Nx}_Ny-{Ny}_k-4.npz', a=eigenvectors)

    try:
        loaded = jnp.load(f'data/states/Nx-{Nx}_Ny-{Ny}_k-4.npz')
        eigenvectors = loaded['a']            
    except:
        eigenvalues, eigenvectors = eigenvalues, eigenvectors = eigsh(H_many_body, k=4, which='SA')
        jnp.savez(f'data/states/Nx-{Nx}_Ny-{Ny}_k-4.npz', a=eigenvectors)
    
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
            eigenvectors = eigenvectors[:,:ground_state_degeneracy].T
        else:
            eigenvectors = None
            
        config_dict['ground_states'] = eigenvectors


        ansatz = Jax_ansatz(Nx = Nx, Ny = Ny, n=n, local_layers_num=config_dict['layer_numer'], flux_gate_true=config_dict['flux_gate_true'])
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
    return res