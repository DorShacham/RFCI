#%% 
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
from tqdm import tqdm
from IQH_state import *
from flux_attch import *

# %%
Nx = 2
Ny = 2
N = Nx * Ny

H_real_space = build_H(Nx, Ny)

eig_val, eig_vec = np.linalg.eigh(H_real_space)
# eigen states (projected on the lower energies) tensor (state index, real space position with A,B sublattices. 
# for position x,y sublattice A the index is 2 * (Ny * x + y) + A
n = N
eigen_states = eig_vec[:,:n].T

# creating the multi particle integer quantum hall state with 2 * N sites and n=N electron (half filled)
mps = Multi_particle_state(2 * N, n)
multi_particle_state_vector = mps.create(eigen_states)

new_mp_state = mps.H_manby_body(H_real_space,multi_particle_state_vector)
# sanity check
print((new_mp_state[np.abs(new_mp_state)>1e-8]/multi_particle_state_vector[np.abs(multi_particle_state_vector)>1e-8]).real)
print(np.linalg.norm(normalize(new_mp_state) + multi_particle_state_vector))
print(np.linalg.norm(multi_particle_state_vector))
print(np.linalg.norm(new_mp_state))
print_mp_state(multi_particle_state_vector, Nx = Nx, Ny = Ny, mps = mps)
# print_mp_state(new_mp_state, Nx = Nx, Ny = Ny, mps = mps)
print_mp_state(mps.time_evolve(H_real_space,multi_particle_state_vector,t=100), Nx = Nx, Ny = Ny, mps = mps)

#%% 
# Exatend the system on the x axis by adding unequippied cites. @extention_factor is a positive integer describe 
# how many cites to add i.e N_new = extention_factor * N

extention_factor = 3
new_N = 2 * (extention_factor * Nx) * Ny   
extended_mps = Multi_particle_state(N = new_N, n = n)
extended_state = extended_mps.zero_vector()

state = multi_particle_state_vector

for index in range(len(state)):
    perm = mps.index_2_perm(index)
    new_perm = []
    for i in range(n):
        cite_index = perm[i]
        x = cite_index // (2 * Ny)
        new_cite_index = cite_index + 2 * Ny * (extention_factor - 1) * x
        new_perm.append(new_cite_index)
    
    assert(permutation_parity(perm,False) == permutation_parity(new_perm,False))
    new_index = extended_mps.perm_2_index(new_perm)
    extended_state[new_index] = state[index]

state = create_IQH_in_extendend_lattice(Nx, Ny, extention_factor)
print(np.linalg.norm(extended_state - state))
#%%
Nx = 2
Ny = 2
# number of electrons - half of the original system
n = Nx * Ny 
extention_factor = 3

extended_state, extended_mps = create_IQH_in_extendend_lattice(Nx = Nx, Ny = Ny, extention_factor = extention_factor)
Nx = extention_factor * Nx
N = 2 * Nx * Ny

extended_H = build_H(Nx = Nx, Ny = Ny)
new_extended_state = extended_mps.H_manby_body(extended_H,extended_state)

# sanity check
# print((new_extended_state[np.abs(new_extended_state)>1e-8]/extended_state[np.abs(extended_state)>1e-8]).real)
print(np.linalg.norm(normalize(new_extended_state) + extended_state))
print(np.linalg.norm(extended_state))
print(np.linalg.norm(new_extended_state))
print_mp_state(extended_state, Nx = Nx, Ny = Ny, mps = extended_mps)
print_mp_state(new_extended_state, Nx = Nx, Ny = Ny, mps = extended_mps)
#%%
promted_state = extended_mps.time_evolve(extended_H, extended_state, t=100)
print(np.linalg.norm(promted_state))
print_mp_state(promted_state, Nx = Nx, Ny = Ny, mps = extended_mps)

#%% 
Nx = 3
Ny = 3
N = 2 * Nx * Ny
n = 3
mps2 = Multi_particle_state(N,n)
H = build_H(Nx,Ny)
states = np.eye(N)[:,:n].T
# tmp = states[0]
# states[0] = states[1]
# states[1] = np.array([1., 0., 0., 0., 0.])
state = mps2.create(states)
print_mp_state(state, Nx = Nx, Ny = Ny, mps = mps2)
print_mp_state(mps2.time_evolve(H_sb=H,multi_particle_state=state,t=100), Nx = Nx, Ny = Ny, mps = mps2)
# %%
#%%
Nx = 2
Ny = 2
# number of electrons - half of the original system
n = Nx * Ny 
extention_factor = 3


state, mps = create_IQH_in_extendend_lattice(Nx = Nx, Ny = Ny, extention_factor = extention_factor)
Nx = extention_factor * Nx
N = 2 * Nx * Ny
state = flux_attch_2_compact_state(state, mps, Ny)


H = build_H(Nx = Nx, Ny = Ny)
# new_state = mps.H_manby_body(H,state)  
# print((new_state[np.abs(new_state)>1e-8]/state[np.abs(state)>1e-8]).real)
# print(np.linalg.norm(normalize(new_state) + state))
# print(np.linalg.norm(state))
# print(np.linalg.norm(new_state))
# print_mp_state(state, Nx = Nx, Ny = Ny, mps = mps)
# print_mp_state(mps.time_evolve(H,state,t=100), Nx = Nx, Ny = Ny, mps = mps)

m = project_on_band(state = state, band = -1, H = H, mps = mps)
print("\n\n")
p = project_on_band(state = state, band = 1, H = H, mps = mps)

print(f"\nAll in all there are:{p + m} electrons\n")
print(f"All in all the energy is:{p - m}")
# %%
#%%
# # TODO
# 1. Write a non-interacting many body hamiltonian and check if multi_particle_state_vector is indeed eigen state with the proper energy
# 2. Write a code that creat this state on Qiskit, first with trivial coding (all many body state of the hilbert space) then with constraied hilbert space