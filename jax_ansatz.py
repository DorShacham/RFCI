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

from IQH_state import Multi_particle_state, permutation_parity
from IQH_state import print_mp_state


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



# A class that act with a translation invariante ansatz on a lattice of (@Nx,@Ny) with 2 sublattice cites and @n electrons
class Jax_TV_ansatz:
    def __init__(self, Nx, Ny, n):
        bonds = [(0,0,1), (0,1,1), (-1,0,1), (-1,1,1)] # NN of the model
        mps = Multi_particle_state(2 * Nx * Ny, n)
        state_size = len(mps.zero_vector())
        
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

                        
        self.state_size = state_size
        self.data_list = data_list
        self.row_indices_list = row_indices_list
        self.col_indices_list = col_indices_list
        


    # appy the ansatz on the @state and return the new_state
    # param_set - the parametrs that parametrize the ansatz, assume to take this form:
    # param_set = [[bond_1_params], [bond_2_params], [bond_3_params], [bond_4_params]], where [bond_n_params] = [a,b,c,d,e,f] real numbers
    @jit
    def apply_ansatz(self, state, param_set):
        num_bonds = 4
        new_state = jnp.array(state, dtype=complex)
        for i, (data,row_indices,col_indices) in enumerate(zip(self.data_list, self.row_indices_list, self.col_indices_list)):
            new_state = apply_single_matrix(new_state,data,row_indices,col_indices,param_set[i // num_bonds])
        return new_state

# Usage
Nx = 2
Ny = 6
N = Nx * Ny
n = 2 * N // 6
mps = Multi_particle_state(2 * Nx * Ny, n)

state = jnp.array(mps.zero_vector()).at[0].set(1)
params = jnp.array([[1., 2., 3., 4., 5., 6.]] * 4)

ansatz = Jax_TV_ansatz(Nx, Ny, n)

new_state2 = ansatz.apply_ansatz(state,params)

print(jnp.linalg.norm(state))
print(jnp.linalg.norm(new_state2))

# print_mp_state(state,Nx,Ny,mps)
# print_mp_state(new_state2,Nx,Ny,mps)

