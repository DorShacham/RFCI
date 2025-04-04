#%%
from functools import partial
import jax
from jax import config
from qiskit_algorithms.optimizers import SPSA
import jaxopt 



# Enable 64-bit computation
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle
import wandb
# from jax.scipy.optimize import approx_fprime
from tqdm import tqdm

from IQH_state import *

def wandb_log_1(config,overlap):
    wandb.log({f'Overlap_{config}': overlap}, commit=False)
def wandb_log_2(config,energy,cost,iter):
    wandb.log(
    {
        f'Energy_config_{config}': energy,
        f'Cost_config_{config}': cost,
        f'step': iter
    })

class Optimizer_reuslt:
    def __init__(self,function_value, x):
        self.function_value = function_value
        self.x = x

class VQE:
    def __init__(self,config):
        self.config = config
        self.initial_state = config['initial_state']
        self.ansatz = config['ansatz']
        self.hamiltonian = config['hamiltonian']
        self.cost_history_dict = {
            "prev_vector": None,
            "iters": 0,
            "cost_history": [],
        }
        self.res = None
        self.path = config['saveto']
        self.log = config['log']
        self.ground_states = config['ground_states']

        if config['loss'] is None:
            self.loss = lambda x:  x
        else:
            self.loss = config['loss']

        self.config_i = config['config_i']
        
        if 'resume_run' in config:
            self.resume_run  = config['resume_run']
        else:
            self.resume_run = False
        if self.resume_run:
            self.cost_history_dict = config['cost_history_dict']


        # jax.debug.print("Output value: {y}", y=sbuspace_probability(state_vector = self.initial_state, subspace = self.ground_states))

        

# calculate the cost_function - the expection value of self.hamiltonian according to self.estimator
# with self.ansatz(@params)
    # @jax.jit
    def cost_func(self,params, for_grad = False):
        state = self.ansatz.apply_ansatz(params = params, state = self.initial_state)
        energy = my_estimator(state, self.hamiltonian)
        cost = energy

        if (not for_grad):
            self.cost_history_dict["iters"] += 1
            # self.cost_history_dict["step_size"] = jnp.linalg.norm(params - self.cost_history_dict["prev_vector"])
            self.cost_history_dict["step_size"] = 1
            self.cost_history_dict["prev_vector"] = params
            self.cost_history_dict["cost_history"].append(energy)

            if (self.config["cktp_iters"] is not None) and  (self.cost_history_dict["iters"] % self.config["cktp_iters"] == 0) and self.log:
                # printing and saving result every cktp
                # print_mp_state(state, self.config['Nx'], self.config['Ny'], self.config['mps'], saveto=str(self.path) + str(f'/electron_density_{self.cost_history_dict["iters"]}.jpg'))
                # wandb.log({"Electron Density": wandb.Image(str(self.path) + str(f'/electron_density_{self.cost_history_dict["iters"]}.jpg'), caption=f"Config {self.config_i} iter {self.cost_history_dict['iters']}")}, commit = False) 
                if self.path is not None:
                    with open(str(self.path) + str('/cktp.pickle'), 'wb') as handle:
                        pickle.dump(self.cost_history_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if self.ground_states is not None:
            overlap = sbuspace_probability(state,subspace=self.ground_states)
            if (not for_grad):
                # wandb.log({f'Overlap_{self.config_i}': (overlap).item()}, commit=False)
                jax.debug.callback(wandb_log_1,self.config_i, overlap)
            if self.config['overlap_optimization']:
                cost  = -overlap

        if (not for_grad) and self.log:
            # wandb.log(
            #     {
            #         f'Energy_config_{self.config_i}': energy,
            #         f'Cost_config_{self.config_i}': cost,
            #         f'step': self.cost_history_dict['iters']
            #     }
            jax.debug.callback(wandb_log_2,self.config_i,energy,cost,self.cost_history_dict['iters'])
        elif not for_grad:
            print(f"Iters. done: {self.cost_history_dict['iters']} [Current cost: {cost}, energy:{energy}]")

        return (self.loss(cost))

    def jacobian(self):
        cost_for_grad = partial(self.cost_func, for_grad = True)
        # print(type(jacfwd(cost_for_grad,(0))(jnp.zeros(self.ansatz.num_parameters()))))
        return (jacfwd(cost_for_grad,(0)))
         

# start the optimization proccess. all data on optimization is saved in self.cost_history_dict
    def minimize(self):
        if not self.resume_run:
            if self.config['random_initial_parametrs']:
                key = jax.random.PRNGKey(0)  # Initialize a random key
                x0 = jnp.array(2 * jnp.pi * jax.random.uniform(key, shape=(self.ansatz.num_parameters(),),dtype=jnp.float64)) * 1e-1
            else:
                x0 = jnp.zeros(shape=(self.ansatz.num_parameters(),))

            if self.config['flux_gate_true']:
                init_flux_params = self.ansatz.flux_gate.get_inital_params()
                x0 = x0.at[:self.ansatz.flux_gate.num_parameters()].set(init_flux_params)
            
            self.cost_history_dict["prev_vector"] = x0
            step_size = 1
        else:
            x0 = self.cost_history_dict["prev_vector"]
            step_size = 1.1 * self.cost_history_dict["step_size"]
        # x0 = x0.astype(jnp.float64)


        if self.config['optimizer'] == 'SPSA':
            spsa = SPSA(maxiter=int(1e7))
            res = spsa.minimize(self.cost_func, x0=x0)
        else:
            # solver = jaxopt.GradientDescent(
            solver = jaxopt.ScipyMinimize(
                method=self.config['optimizer'],
                fun=self.cost_func,
                maxiter=self.config['maxiter'],
                tol= 1e-5,
                options = {
                'gtol': 1e-15,  # Gradient tolerance
                },
                dtype=jnp.float64
                )
            res = solver.run(init_params=x0)
            # res = minimize(
            #     self.cost_func,
            #     x0,
            #     args=(),
            #     # method=self.config['optimizer'],
            #     # method='l-bfgs-experimental-do-not-rely-on-this',
            #     method="BFGS",
            #     # jac = self.jacobian(),
            #     # method="cobyla",
            #     # method="SLSQP",
            #     tol=1e-8,
            #     # options={"maxiter":self.config['maxiter'], "rhobeg":0.1},
            #     options = {
            #     'gtol': 1e-15,  # Gradient tolerance
            #     # 'eps': 1e-08,  # Step size for numerical approximation
            #     'maxiter': self.config['maxiter'],  # Maximum iterations,
            #     # 'line_search_maxiter': 50,
            #     # 'maxls': 1000,
            #     },
            #     # options={"maxiter":self.config['maxiter']},
            #     # bounds= [(0, 2*np.pi) for _ in range(self.ansatz.num_parameters)]
            # )

        print(res.state)
        self.res = res
        if self.path is not None:
            with open(str(self.path) + str('/res.pickle'), 'wb') as handle:
                pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if self.log:
                artifact = wandb.Artifact(f'model_weights_config_{self.config_i}', type='model')
                artifact.add_file(str(self.path) + str('/res.pickle'))
                # Log the artifact
                wandb.log_artifact(artifact)

        return res

# plot cost function as a function of iterations.
    def plot(self):
        plt.figure()
        plt.plot(range(self.cost_history_dict["iters"]), self.cost_history_dict["cost_history"])
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.draw()
        if self.path is not None:
            plt.savefig(str(self.path) + str('/optimzation_plot.jpg'))
        plt.close()


# Calculate the expection value of @operator on final state from @qc with @initial_state
@jax.jit
def my_estimator(state,operator):
    return ((state.T.conjugate() @ (operator @ state)) / (state.T.conjugate() @ state)).real

# Calculate the probatility of @state_vector to be in the subspace spanned by the set @subspace
@jax.jit
def sbuspace_probability(state_vector, subspace):
    prob = 0
    for v in subspace:
        prob += jnp.abs(state_vector.T.conjugate() @ v) ** 2
    # jax.debug.print("Output value: {y}", y=prob)
    return prob

