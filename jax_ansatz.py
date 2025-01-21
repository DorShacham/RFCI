#%%
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


from IQH_state import *

# def M(data):
#     data_indices = [0,0,1]
#     data = jnp.array([data[i] for i in data_indices])
#     sparse_matrix = sparse.BCOO((data, indices), shape=(3, 3))
#     return sparse_matrix



# vector = jnp.array([1., 2., 3.])
# result = (M([1,2]) @ vector)

# print(result)

# implimenting single term in translation invaritant ansatz. 
# @state is a multi-particle compact state with @mps
# this ansatz will couble between state with electron in @cite_index_1 and @cite_index_2
# parmas = (a,b,c,d,e,f) real vector from which the ansatz will be built
# @return the state after acting on it with the ansatz
def single_term_ansatz(state,mps,cite_index_1,cite_index_2,params):
    assert(not (cite_index_1 == cite_index_2)) # no self coupling
    a,b,c,d,e,f = params
    phase1 = np.exp(1j * e)
    phase2 = np.exp(1j * f)
    hermition_matrix = jnp.array([[a, b - 1j * c], [b + 1j *c, d]])
    unitary_matrix = jax.scipy.linalg.expm(1j * hermition_matrix)
    new_state = mps.zero_vector()

    for index in range(len(new_state)):
            state_perm = mps.index_2_perm(index)
            if (not (cite_index_1 in state_perm)) and (not (cite_index_2 in state_perm)):
                new_state[index] =  phase1 * state[index]
                continue
            elif ((cite_index_1 in state_perm)) and ((cite_index_2 in state_perm)):
                new_state[index] =  phase2 * state[index]
                continue
            else:
                cite_sum = cite_index_1 + cite_index_2
                for j, cite_index in (enumerate([cite_index_1,cite_index_2])):
                    if cite_index in state_perm:
                        # C_2_dagger (U10) C_1 |state> / # C_1_dagger (U01) C_2 |state>
                        
                        # find index of C_j
                        k = state_perm.index(cite_index) 
                        # contract C1 with C1_dagger and insert C2_dagger to the right place 
                        new_perm =  list(state_perm)
                        del new_perm[k]
                        new_perm.insert(0,(cite_sum - cite_index))
                        parity, sorted_perm = permutation_parity(tuple(new_perm), return_sorted_array=True)
                        new_index = mps.perm_2_index(sorted_perm)
                        
                        # C_2_dagger (U10) C_1 |state>
                        new_state[index] += unitary_matrix[j,j] * state[index]
                        new_state[new_index] += unitary_matrix[(1-j),j] * (-1)**k * (-1)**parity * state[index]
                        continue


    return new_state

# implimenting a lattice translation invaritant ansatz for a given @bond
# @state is a multi-particle compact state with @mps for lattice with @lattice_shape = (Nx,Ny,sublattice)
# this ansatz will couble between state with electron in cite_index_i and cite_index_j that will be coubled by 
# @bond =  (delta_x, delta_y, delta_sublattice)
# parmas = (a,b,c,d,e,f) real vector from which the ansatz will be built
# @return the state after acting on it with the ansatz
def bond_ansatz_terms(state,mps,lattice_shape,bond,params):
    new_state = state
    Nx, Ny, sublattice = lattice_shape
    for x in range(Nx):
        for y in range(Ny):
            #cite_index = 2 * (Ny * x + y) + sublattice
            cite_index_1 = 2 * (Ny * x + y) + 0
            x2 = (x + bond[0]) % Nx
            y2 = (y + bond[1]) % Ny
            sublattice2 = 0 + bond[2]
            cite_index_2 = 2 * (Ny * x2 + y2) + sublattice2

            new_state = single_term_ansatz(new_state, mps, cite_index_1, cite_index_2, params)


    return new_state

# implimenting a lattice translation invaritant ansatz.
# @state is a multi-particle compact state with @mps for lattice with @lattice_shape = (Nx,Ny,sublattice)
# @parmas_set - a set of 6 tuples of params=(a,b,c,d,e,f) real vector from which the ansatz will be built
# @return the state after acting on it with the ansatz
def translation_invariant_ansatz(state,mps,lattice_shape,params_set):
    new_state = state
    new_state = bond_ansatz_terms(new_state,mps,lattice_shape,bond=(0,0,1),params=params_set[0])
    new_state = bond_ansatz_terms(new_state,mps,lattice_shape,bond=(0,-1,1),params=params_set[1])
    new_state = bond_ansatz_terms(new_state,mps,lattice_shape,bond=(-1,0,1),params=params_set[2])
    new_state = bond_ansatz_terms(new_state,mps,lattice_shape,bond=(-1,-1,1),params=params_set[3])

    return new_state



#%%

Nx = 2
Ny = 6
sublattice = 2
lattice_shape = (Nx,Ny,sublattice)

N = Nx * Ny
n = 2 * N // 6
mps = Multi_particle_state(2 * Nx * Ny, n)
# state = mps.zero_vector()
# state += 1
# state = state / np.linalg.norm(state)
# state[0] = 1
H_real_space = build_H(Nx,Ny)
IQH_state, mps = create_IQH_in_extendend_lattice(Nx,Ny,n,extention_factor = 1, band_energy = 1, H_sb = H_real_space)
state = IQH_state
print_mp_state(state,Nx,Ny,mps)


param_set = [(1,2,3,4,5,6)] * 4
# param_set = [(0,2,0,0,0,0)] * 4
new_state = translation_invariant_ansatz(state,mps,lattice_shape, param_set)

print(np.linalg.norm(state))
print(np.linalg.norm(new_state))

print_mp_state(new_state,Nx,Ny,mps)


#%%
import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

def create_ansatz_matrix(mps, lattice_shape, bonds):
    Nx, Ny, _ = lattice_shape
    n_sites = 2 * Nx * Ny
    state_size = len(mps.zero_vector())

    data = []
    row_indices = []
    col_indices = []

    for bond in bonds:
        for x in range(Nx):
            for y in range(Ny):
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

    return jnp.array(data, dtype=np.float32), (row_indices), (col_indices), state_size

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
def apply_ansatz(state, data, row_indices, col_indices, matrix_shape, params):
    values = compute_matrix_values(data, params)
    
    # Custom sparse matrix-vector multiplication
    def sparse_mv(v):
        return jax.ops.segment_sum(
            values * v[col_indices],
            row_indices,
            matrix_shape[0]
        )
    
    return sparse_mv(state)

# Usage
Nx, Ny, sublattice = 2, 6, 2
lattice_shape = (Nx, Ny, sublattice)
N = Nx * Ny
n = 2 * N // 6
mps = Multi_particle_state(2 * Nx * Ny, n)
bonds = [(0,0,1), (0,1,1), (-1,0,1), (-1,-1,1)]

data, row_indices, col_indices, state_size = create_ansatz_matrix(mps, lattice_shape, bonds)
matrix_shape = (state_size, state_size)

# Convert to JAX arrays
data = jnp.array(data, dtype=jnp.float32)
row_indices = jnp.array(row_indices, dtype=jnp.int32)
col_indices = jnp.array(col_indices, dtype=jnp.int32)
matrix_shape = jnp.array(matrix_shape, dtype=jnp.int32)

state = jnp.zeros(state_size).at[0].set(1)
params = jnp.array((1., 2., 3., 4., 5., 6.))
new_state = apply_ansatz(state, data, row_indices, col_indices, matrix_shape, params)

print(jnp.linalg.norm(state))
print(jnp.linalg.norm(new_state))

