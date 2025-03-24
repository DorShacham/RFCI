#%%
import jax
from jax import config

# Enable 64-bit computation
config.update("jax_enable_x64", True)
import jax.numpy as jnp


from IQH_state import *
from flux_attch import *
# import numpy as np
from jax_ansatz import *
import pickle
from chern_ins_magnetic import *
from jax_vqe import sbuspace_probability

p = -1
q = 3
Nx = 3
Ny = 6
n = Nx * Ny // 3
mps = Multi_particle_state(2 * Nx * Ny, n)


loaded2 = jnp.load(f'data/states/Nx-{Nx}_Ny-{Ny}_k-4.npz')
eigenvectors = loaded2['a']
state = jnp.load(f'data/states/Nx-{Nx}_Ny-{Ny}_q=3_magnetic.npy')

ansatz = Jax_ansatz(Nx,Ny,n,local_layers_num=1,flux_gate_true=True)
key = jax.random.PRNGKey(0)  # Initialize a random key
x0 = jnp.array(2 * jnp.pi * jax.random.uniform(key, shape=(ansatz.num_parameters(),),dtype=float)) * 1e-1
init_flux_params = ansatz.flux_gate.get_inital_params()
x0 = x0.at[:ansatz.flux_gate.num_parameters()].set(init_flux_params)


v = jnp.array([jnp.array(state),translation_operator(state,mps,Nx,Ny,Tx=0,Ty=1),translation_operator(state,mps,Nx,Ny,Tx=0,Ty=2)])

for a in range(3):
    state = jnp.load(f'data/states/Nx-{Nx}_Ny-{Ny}_q=3_magnetic.npy')
    print(f"-------------\n\na={a}:")
    if True: # state is not symmetric
        # sym state = phase * (phase ** (-1/3) I + phase ** (-2/3) T + phase ** (-3/3) T^2) state
        phase = state.T.conjugate() @ translation_operator(state,mps,Nx,Ny,Tx=0,Ty=3)
        print(f"phase:{phase}")
        sym_state = jnp.array(v[0]) * (phase ** (-1/3)) * jnp.exp(1j * 2 * jnp.pi / (-3) * a)
        for i in range(1,q):
            sym_state += v[i] * (phase ** (-(i+1)/3)) * jnp.exp(1j * 2 * jnp.pi / (-3) * (i+1) * a)
        state = normalize(sym_state * phase)


    state = ansatz.apply_ansatz(params=x0,state=state)
    T_x_expectation = state.T.conjugate() @ translation_operator(state,mps,Nx,Ny,Tx=1,Ty=0)
    T_y_expectation = state.T.conjugate() @ translation_operator(state,mps,Nx,Ny,Tx=0,Ty=1)
    print(jnp.abs(T_x_expectation))
    print(jnp.abs(T_y_expectation))
    # print(jnp.linalg.norm(state - translation_operator(state,mps,Nx,Ny,Tx=0,Ty=3)))
    # print_mp_state(state,Nx,Ny,mps)

    Kx = (jnp.angle(T_x_expectation)  / (2 * jnp.pi) * Nx) % Nx % Nx
    Ky = (jnp.angle(T_y_expectation)  / (2 * jnp.pi) * Ny) % Ny % Ny
    print((Kx,Ky))
    print(Kx + Nx * Ky)
    print(f"subspace probe:{sbuspace_probability(state,subspace=eigenvectors.T)}")