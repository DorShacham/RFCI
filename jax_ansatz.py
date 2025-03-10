#%%
# Jax version - calculting each sprase matrix and in each iteration substitue the parametrs and then compute the multiplaction
import os
import jax

# Set the number of CPU devices JAX will use
jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_default_matmul_precision', 'float32')

# Use all available CPUs
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count={}'.format(os.cpu_count())
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jax import jit, vmap
import numpy as np
from tqdm import tqdm
from flux_attch import *
from mpmath import *


from IQH_state import *

# All the function for the Transliation Invariant Ansatz
@jit
def create_unitary_matrix(params):
    a, b, c, d, _, _ = params
    hermitian_matrix = jnp.array([[a, b - 1j * c], [b + 1j * c, d]])
    return jax.scipy.linalg.expm(1j * hermitian_matrix)

@jit
def compute_matrix_values(data, params):
    a, b, c, d, e, f = params
    unitary_matrix = create_unitary_matrix(params)
    phase1 = jnp.exp(1j * e)
    phase2 = jnp.exp(1j * f)
# j, k, l, switch_index = entry
# @switch_index == 0: U(0,0) / U(1,1) [j,k =0,0 or j,j = 1,1]
# @switch_index == 1: U(0,1) / U(0,1) * (-1)**(k + parity) [j,k = 0,1 or j,k = 1,0, l = k+ parity]
# @switch_index == 2: phase1 / phase2 [j = 1 for phase1 and j = 2 for phase2]
 

    def compute_value(entry):
        j, k, l, switch_index = entry
        return jax.lax.switch(switch_index,
            [
                lambda: unitary_matrix[j.astype(int), k.astype(int)],
                lambda: unitary_matrix[j.astype(int), k.astype(int)] * (-1)**l.astype(int),
                lambda: jnp.where(j == 1, phase1, phase2)
            ]
        )

    return vmap(compute_value)(data)

@jit
def apply_single_matrix(state,data,row_indices,col_indices,params):
    matrix_shape = (len(state), len(state))
    values = compute_matrix_values(data,params)
    indices = jnp.column_stack((row_indices, col_indices))
    jax_sparse_matrix = sparse.BCOO((values,indices),shape = matrix_shape)
    return (jax_sparse_matrix @ state)

@jit
def _apply_TV_ansatz(state, param_set, data_array, row_indices_array, col_indices_array):
    num_bonds = 4

    def scan_body(state, inputs):
        data, row_indices, col_indices, params = inputs
        new_state = apply_single_matrix(state, data, row_indices, col_indices, params)
        return new_state, new_state

    initial_state = jnp.array(state, dtype=complex)
    xs = (data_array, row_indices_array, col_indices_array, param_set)
    final_state, _ = jax.lax.scan(scan_body, initial_state, xs)
    
    return final_state



# A class that act with a translation invariante ansatz on a lattice of (@Nx,@Ny) with 2 sublattice cites and @n electrons
class Jax_TV_ansatz:
    def __init__(self, Nx, Ny, n):
        # NN = []; N = [(0,0), (0,1), (-1,0), (-1,1)]
        bonds = [(0,0,1), (0,1,1), (-1,0,1), (-1,1,1)] # NN of the model
        self.num_bonds = len(bonds)
        self.params_per_bond = 6
        self.Nx = Nx
        self.Ny = Ny
        
        mps = Multi_particle_state(2 * Nx * Ny, n)
        state_size = mps.len
        self.state_size = state_size
        
        try: # trying to load existing matrix
            loaded = np.load(str(f'data/matrix/ansatz/local_gate_Nx-{Nx}_Ny-{Ny}.npz'))
            self.data_array = jnp.array(loaded['data_array'])
            self.row_indices_array = jnp.array(loaded['row_indices_array'])
            self.col_indices_array = jnp.array(loaded['col_indices_array'])
        
        except:
            # building loacal ansatz gate
            print("Building local ansatz gate")
            data_list = []
            row_indices_list = []
            col_indices_list = []

            for bond in bonds:
                for x in (range(Nx)):
                    for y in (range(Ny)):
                        data = []
                        row_indices = []
                        col_indices = []

                        cite_index_1 = 2 * (Ny * x + y)
                        x2 = (x + bond[0]) % Nx
                        y2 = (y + bond[1]) % Ny
                        cite_index_2 = 2 * (Ny * x2 + y2) + bond[2]

                        for index in range(state_size):
                            state_perm = mps.index_2_perm(index)
                            in_cite1 = cite_index_1 in state_perm
                            in_cite2 = cite_index_2 in state_perm

                            if not in_cite1 and not in_cite2:
                                row_indices.append(index)
                                col_indices.append(index)
                                data.append([1, 0, 0, 2])  # Placeholder for phase1 = exp(1j * e)
                            elif in_cite1 and in_cite2:
                                row_indices.append(index)
                                col_indices.append(index)
                                data.append([2, 0, 0, 2])  # Placeholder for phase2 = exp(1j * f)
                            else:
                                cite_sum = cite_index_1 + cite_index_2
                                for j, cite_index in enumerate([cite_index_1, cite_index_2]):
                                    if cite_index in state_perm:
                                        k = state_perm.index(cite_index)
                                        new_perm = list(state_perm)
                                        del new_perm[k]
                                        new_perm.insert(0, cite_sum - cite_index)
                                        parity, sorted_perm = permutation_parity(tuple(new_perm), return_sorted_array=True)
                                        new_index = mps.perm_2_index(sorted_perm)

                                        row_indices.append(index)
                                        col_indices.append(index)
                                        data.append([j, j, 0, 0])  # Placeholder for unitary_matrix[j,j]

                                        row_indices.append(new_index)
                                        col_indices.append(index)
                                        data.append([1-j, j, k + parity, 1])  # Placeholder for unitary_matrix[1-j,j] * (-1)**(k + parity)
                        
                        data_list.append(jnp.array(data, dtype=jnp.int32))
                        row_indices_list.append(jnp.array(row_indices, dtype=jnp.int32))
                        col_indices_list.append(jnp.array(col_indices, dtype=jnp.int32))
                                                    
                        self.data_array = jnp.array(data_list)
                        self.row_indices_array = jnp.array(row_indices_list)
                        self.col_indices_array = jnp.array(col_indices_list)

                        np.savez(str(f'data/matrix/ansatz/local_gate_Nx-{Nx}_Ny-{Ny}.npz'), data_array=self.data_array, row_indices_array=self.row_indices_array, col_indices_array=self.col_indices_array)

        


    # apply the ansatz on the @state and return the new_state
    # param_set - the parametrs that parametrize the ansatz. After ordering takes this form
    # ordered_params = [[bond_1_params], [bond_1_params], ..., [bond_2_params], [bond_2_params], ... [bond_4_params]], 
    # where [bond_n_params] = [a,b,c,d,e,f] real numbers
    def apply_ansatz(self, param_set, state):
        system_size = self.Nx * self.Ny
        ordered_params = [[param_set[i * self.params_per_bond: (i + 1) * self.params_per_bond]] * system_size for i in range(self.num_bonds)]
        ordered_params = [item for sublist in ordered_params for item in sublist]
        return _apply_TV_ansatz(state, jnp.array(ordered_params), self.data_array, self.row_indices_array, self.col_indices_array)


    def num_parameters(self):
        param_number =  self.num_bonds * self.params_per_bond
        return param_number



# #####
# # Usage
# Nx = 2
# Ny = 6
# N = Nx * Ny
# n = 2 * N // 6
# mps = Multi_particle_state(2 * Nx * Ny, n)

# state = jnp.array(mps.zero_vector()).at[0].set(1)
# params = jnp.array([1., 2., 3., 4., 5., 6.] * 4 )

# ansatz = Jax_TV_ansatz(Nx, Ny, n)

# #%%
# import time

# start = time.time()
# for i in range(1000):
#     new_state = ansatz.apply_ansatz(state,params)
# end = time.time()

# print((end - start) / 1000)
# print(jnp.linalg.norm(state))
# print(jnp.linalg.norm(new_state))

# # print_mp_state(state,Nx,Ny,mps)
# # print_mp_state(new_state,Nx,Ny,mps)




###########################################
# All the function and call for the flux attachment ansatz

@jit
def _apply_FA_ansatz(phase_structure_matrix, params,state):
    return jnp.exp(1j * (phase_structure_matrix @ params)) * state

class Jax_FA_ansatz:
    # transliation invariant flux attach ansatz matrix
    def __init__(self,Nx,Ny,IQH_state_mps):
        self.Nx = Nx
        self.Ny = Ny
        self.mps = IQH_state_mps

        try: # try loading exisiting matrix
            loaded = np.load(str(f'data/matrix/ansatz/phase_matrix_Nx-{Nx}_Ny-{Ny}.npz'))
            data = loaded['data']
            indices = loaded['indices']
            shape = tuple(loaded['shape'])

            sparse_matrix = sparse.BCOO((data, indices), shape=shape)
        
        except:
            # building phase addition matrix
            print("Building flux attachment ansatz matrix")
            mps = IQH_state_mps
            cite_number = 2 * Nx * Ny
            state_size = IQH_state_mps.len
            matrix_elements = {}
            for index in range(state_size):
                state_perm = mps.index_2_perm(index)
                for cite1 in range(1, len(state_perm)):
                    for cite2 in range(cite1):
                        # determine shortest vector
                        x1,y1,sublattice1 = cite_index_2_cite(state_perm[cite1],Ny)
                        x2,y2,sublattice2 = cite_index_2_cite(state_perm[cite2],Ny)
                        
                        dist_1 = np.sqrt(((x1 - x2) % Nx)**2 + ((y1 - y2) % Ny)**2)
                        dist_2 = np.sqrt((-(x1 - x2) % Nx)**2 + ((y1 - y2) % Ny)**2)
                        dist_3 = np.sqrt(((x1 - x2) % Nx)**2 + (-(y1 - y2) % Ny)**2)
                        dist_4 = np.sqrt((-(x1 - x2) % Nx)**2 + (-(y1 - y2) % Ny)**2)
                        dist_list = [dist_1, dist_2, dist_3, dist_4]
                        
                        col_dict = {
                            0: cite_2_cite_index(x = ((x1 - x2) % Nx), y = ((y1 - y2) % Ny), sublattice = 0, Ny = Ny),
                            1: cite_2_cite_index(x = (-(x1 - x2) % Nx), y = ((y1 - y2) % Ny), sublattice = 0, Ny = Ny),
                            2: cite_2_cite_index(x = ((x1 - x2) % Nx), y = (-(y1 - y2) % Ny), sublattice = 0, Ny = Ny),
                            3: cite_2_cite_index(x = (-(x1 - x2) % Nx), y = (-(y1 - y2) % Ny), sublattice = 0, Ny = Ny)
                            }
                        col = col_dict[dist_list.index(min(dist_list))]
                        matrix_elements[(index, col)] = matrix_elements.get((index, col), 0)  + 1

            rows, cols = zip(*matrix_elements.keys())
            values = list(matrix_elements.values())
            indices = jnp.column_stack((jnp.array(rows), jnp.array(cols)))
            sparse_matrix = sparse.BCOO((values,indices),shape = (state_size, cite_number))

            np.savez(str(f'data/matrix/ansatz/phase_matrix_Nx-{Nx}_Ny-{Ny}.npz'), data=sparse_matrix.data, indices=sparse_matrix.indices, shape=sparse_matrix.shape)
            
        self.phase_structure_matrix = sparse_matrix

    # original phase structure with an element for each pair of electron - most general
    def most_general__init__(self,Nx,Ny,IQH_state_mps):
        self.Nx = Nx
        self.Ny = Ny
        self.mps = IQH_state_mps

        # building phase addition matrix
        mps = IQH_state_mps
        cite_number = 2 * Nx * Ny
        mps_2_particles = Multi_particle_state(N=cite_number,n=2)
        state_size = mps.len
        matrix_elements = {}
        # A = jnp.zeros((mps.len),len(mps_2_particles.zero_vector())))
        for index in range(state_size):
            state_perm = mps.index_2_perm(index)
            for cite1 in range(1, len(state_perm)):
                for cite2 in range(cite1):
                    matrix_elements[(index, mps_2_particles.perm_2_index((state_perm[cite2],state_perm[cite1])))] = 1
                    # A = A.at[[index, mps_2_particles.perm_2_index((state_perm[cite2],state_perm[cite1]))]].set(1)

        rows, cols = zip(*matrix_elements.keys())
        values = list(matrix_elements.values())
        indices = jnp.column_stack((jnp.array(rows), jnp.array(cols)))
        sparse_matrix = sparse.BCOO((values,indices),shape = (state_size, len(mps_2_particles.zero_vector())))
        # sparse_matrix = sparse.csr_matrix((values, (rows, cols)))
        # sparse_matrix = jax_sparse.BCOO.from_scipy_sparse(sparse_matrix)
        
        self.phase_structure_matrix = sparse_matrix

    def apply_ansatz(self, params,state):
        return _apply_FA_ansatz(self.phase_structure_matrix, params, state)

# for the TA flux ansatx
    def get_inital_params(self):
        Nx = self.Nx
        Ny = self.Ny
        cite_number = 2 * Nx * Ny
        init_params = np.zeros((cite_number), dtype=jnp.float64)

        for cite in range(cite_number):
            x,y,sublattice = cite_index_2_cite(cite,Ny)
            dist_1 = np.sqrt((x % Nx)**2 + (y % Ny)**2)
            dist_2 = np.sqrt((-x % Nx)**2 + (y % Ny)**2)
            dist_3 = np.sqrt((x % Nx)**2 + (-y % Ny)**2)
            dist_4 = np.sqrt((-x % Nx)**2 + (-y % Ny)**2)
            dist_list = [dist_1, dist_2, dist_3, dist_4]
            
            col_dict = {
                0: cite_2_cite_index(x = (x % Nx), y = (y % Ny), sublattice = 0, Ny = Ny),
                1: cite_2_cite_index(x = (-x % Nx), y = (y % Ny), sublattice = 0, Ny = Ny),
                2: cite_2_cite_index(x = (x % Nx), y = (-y % Ny), sublattice = 0, Ny = Ny),
                3: cite_2_cite_index(x = (-x % Nx), y = (-y % Ny), sublattice = 0, Ny = Ny)
                }
            shortest_dist_cite = col_dict[dist_list.index(min(dist_list))]

            za = cite_index_2_z(shortest_dist_cite, self.mps, self.Ny)
            z = (za) / self.Nx
            tau = 1j *self.Ny / self.Nx
            q = complex(np.exp(1j *jnp.pi * tau))
            term = jnp.array(complex(jtheta(1,z,q)**2))
            if jnp.abs(term) > 1e-6:
                init_params[shortest_dist_cite] = float(np.angle(term))

        return init_params

# for the general flux ansatx
    def general_get_inital_params(self):
        cite_number = 2 * self.Nx * self.Ny
        mps_2_particles = Multi_particle_state(N=cite_number,n=2)
        
        init_params = mps_2_particles.zero_vector().astype(float)

        for a in range(1 , cite_number):
            for b in range(a):
                za = cite_index_2_z(a, self.mps, self.Ny)
                zb = cite_index_2_z(b, self.mps, self.Ny)
                z = (zb - za) / self.Nx
                tau = 1j *self.Ny / self.Nx
                q = complex(np.exp(1j *jnp.pi * tau))
                term = jnp.array(complex(jtheta(1,z,q)**2))
                if jnp.abs(term) > 1e-6:
                    init_params[mps_2_particles.perm_2_index((b,a))] = float(np.angle(term))

        return init_params

    def num_parameters(self):
        row, col = jnp.shape(self.phase_structure_matrix)
        return col
        

    def assign_parameters(self, params):
        operate = partial(self.operate,params=params)
        return operate


########
# A class that unify the complete jax ansatz
# made of U_ansatz = U_local_after @ U_flux @ U_local_before

class Jax_ansatz:
    def __init__(self,Nx,Ny,n, local_layers_num = 1, flux_gate_true = True):
        IQH_state_mps =  Multi_particle_state(2 * Nx * Ny, n)
        self.flux_gate_true = flux_gate_true
        
        if flux_gate_true:
            self.flux_gate = Jax_FA_ansatz(Nx = Nx, Ny = Ny, IQH_state_mps = IQH_state_mps)
            self.flux_gate_param_num = self.flux_gate.num_parameters()
        else:
            self.flux_gate = None
            self.flux_gate_param_num = 0

        self.local_gate = Jax_TV_ansatz(Nx = Nx, Ny = Ny, n = n)
        self.local_gate_param_num = self.local_gate.num_parameters()
        self.local_layers_num = local_layers_num
    
    # The first flux_gate_param_num is for the flux gate
    # after that local_gate_param_num for the local before gate 
    # after that local_gate_param_num for the local after gate
    def apply_ansatz(self, params, state):
        param_list = list(params)
        flux_params = [param_list.pop(0) for _ in range(self.flux_gate_param_num)]
        local_before = []
        local_after = []
        for i in range(self.local_layers_num):
            local_before.append([param_list.pop(0) for _ in range(self.local_gate_param_num)])
            local_after.append([param_list.pop(0) for _ in range(self.local_gate_param_num)])

        state= jnp.array(state)
        for before_params in local_before:
            state = self.local_gate.apply_ansatz(param_set=jnp.array(before_params), state= state)

        if self.flux_gate_true:
            state = self.flux_gate.apply_ansatz(params=jnp.array(flux_params), state=state)
            
        for after_params in local_after:
            state = self.local_gate.apply_ansatz(param_set=jnp.array(after_params), state= state)
        return state

    def num_parameters(self):
        return (self.flux_gate_param_num + self.local_gate_param_num * 2 * self.local_layers_num)

#%%
# import jax.random as random
# key = random.PRNGKey(0)

# Nx = 3
# Ny = 4
# n = Nx * Ny // 3
# ansatz = Jax_ansatz(Nx,Ny,n)
# # state = jnp.zeros(ansatz.local_gate.state_size).at[0].set(1)
# # mps = Multi_particle_state(2 * Nx * Ny, n)
# state, mps = create_IQH_in_extendend_lattice(Nx,Ny, n)
# state = jnp.array(state)
# #%%
# key = random.PRNGKey(9)
# random_params = random.uniform(key, shape=(ansatz.num_parameters(),))
# # random_params = random_params.at[:ansatz.flux_gate.num_parameters()].set(ansatz.flux_gate.get_inital_params())
# new_state = ansatz.apply_ansatz(random_params,state)

# print_mp_state(state,Nx,Ny,mps)
# print_mp_state(new_state,Nx,Ny,mps)