#%%
from scipy.linalg import expm
import os
import jax
import jax.numpy as np
from  jax.experimental import sparse as jax_sparse
from  scipy import sparse 


from IQH_state import *
from flux_attch import *
from jax_vqe import *
from exact_diagnolization import exact_diagnolization



class Ansatz_class:
    def __init__(self,Nx,Ny,IQH_state_mps):
        self.Nx = Nx
        self.Ny = Ny

        # building phase addition matrix
        mps = IQH_state_mps
        cite_number = 2 * Nx * Ny
        mps_2_particles = Multi_particle_state(N=cite_number,n=2)
        matrix_elements = {}
        # A = np.zeros((len(mps.zero_vector()),len(mps_2_particles.zero_vector())))
        for index in range(len(mps.zero_vector())):
            state_perm = mps.index_2_perm(index)
            for cite1 in range(1, len(state_perm)):
                for cite2 in range(cite1):
                    matrix_elements[(index, mps_2_particles.perm_2_index((state_perm[cite2],state_perm[cite1])))] = 1
                    # A = A.at[[index, mps_2_particles.perm_2_index((state_perm[cite2],state_perm[cite1]))]].set(1)

        rows, cols = zip(*matrix_elements.keys())
        values = list(matrix_elements.values())
        sparse_matrix = sparse.csr_matrix((values, (rows, cols)))
        sparse_matrix = jax_sparse.BCOO.from_scipy_sparse(sparse_matrix)
        
        self.phase_structure_matrix = sparse_matrix

    def flux_gate(self, params,state):
        return np.exp(1j * (self.phase_structure_matrix @ params)) * state


    def num_parameters(self):
        row, col = np.shape(self.phase_structure_matrix)
        return col

    def operate(self, params, state):
        returned_state = np.array(state)
        returned_state = self.flux_gate(params=params,state=returned_state)
        return returned_state
        

    def assign_parameters(self, params):
        operate = partial(self.operate,params=params)
        return operate


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
        IQH_state = np.load(f'data/Nx-{Nx}_Ny-{Ny}_q={q}_magnetic.npy')
        mps = Multi_particle_state(N = 2 * Nx * Ny,n=n)
    except:
        print("Calculting IQH state")
        H_real_space = build_H(Nx,Ny)
        H_real_space_magnetic = add_magnetic_field(np.array(H_real_space), p, q, Nx, Ny, cites_per_uc = 2)
        IQH_state, mps = create_IQH_in_extendend_lattice(Nx,Ny,n,extention_factor = 1, band_energy = 1, H_sb = H_real_space_magnetic)
        np.save(f'data/Nx-{Nx}_Ny-{Ny}_q={q}_magnetic',IQH_state)

    state = IQH_state
    H_many_body = jax_sparse.BCOO.from_scipy_sparse(sparse.load_npz(f'results/Exact_Diagnolization/Nx-{Nx}_Ny-{Ny}/sparse_matrix.npz'))

    for i, config_dict in enumerate(config_list):
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
                loaded = np.load(f'data/Nx-{Nx}_Ny-{Ny}_k-4.npz')
                eigenvectors = loaded['a']            
                eigenvectors = eigenvectors
            except:
                eigenvalues, eigenvectors = exact_diagnolization(Nx, Ny, band_energy=config_dict['band_energy'], interaction_strength=config_dict['interaction_strength'],k=config_dict['ground_state_degeneracy'],multi_process=False, save_result=False, show_result=False)
                eigenvectors = eigenvectors
            eigenvectors = eigenvectors[:,:ground_state_degeneracy].T
        else:
            eigenvectors = None
            
        config_dict['ground_states'] = eigenvectors


        ansatz = Ansatz_class(Nx = Nx, Ny = Ny, IQH_state_mps=mps)
        config_dict['ansatz'] = ansatz
        
        vqe = VQE(config_dict)
        res = vqe.minimize()
        vqe.plot()
        # calculting initial and final energy
        i_state = state
        f_state = ansatz.operate(params = res.x, state = state)

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
