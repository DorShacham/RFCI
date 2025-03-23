#%%
from functools import partial
import jax
from jax import config

# Enable 64-bit computation
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.numpy as jnp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle
import wandb
# from jax.scipy.optimize import approx_fprime
from tqdm import tqdm

from IQH_state import *

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
            self.loss = lambda x: x
        else:
            self.loss = config['loss']

        self.config_i = config['config_i']
        
        if 'resume_run' in config:
            self.resume_run  = config['resume_run']
        else:
            self.resume_run = False
        if self.resume_run:
            self.cost_history_dict = config['cost_history_dict']

        

# calculate the cost_function - the expection value of self.hamiltonian according to self.estimator
# with self.ansatz(@params)
    # @jax.jit
    def cost_func(self,params, for_grad = False):
        state = self.ansatz.apply_ansatz(params = params, state = self.initial_state)
        energy = my_estimator(state, self.hamiltonian)
        cost = energy

        if (not for_grad):
            self.cost_history_dict["iters"] += 1
            self.cost_history_dict["step_size"] = np.linalg.norm(params - self.cost_history_dict["prev_vector"])
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
                wandb.log({f'Overlap_{self.config_i}': overlap}, commit=False)
            if self.config['overlap_optimization']:
                cost  = -overlap

        if (not for_grad) and self.log:
            wandb.log(
                {
                    f'Energy_config_{self.config_i}': energy,
                    f'Cost_config_{self.config_i}': cost,
                    f'step': self.cost_history_dict['iters']
                }
            )
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
            spsa = SPSA(maxiter=300)
            res = spsa.minimize(self.cost_func, x0=x0)
        elif self.config['optimizer'] == 'my_optimizer':
            res = my_optimizer(x = x0,cost_func=self.my_cost_func ,eps=1e-8, step_size=1e-3, approx_min=self.approx_min, log= self.log)
        elif self.config['optimizer'] == 'my_optimizer_V2':
            res = my_optimizer_V2(x = x0,cost_func=self.my_cost_func ,eps=1e-8, step_size=1, approx_min=self.approx_min, log= self.log)
        else:
            res = minimize(
                self.cost_func,
                x0,
                args=(),
                method=self.config['optimizer'],
                # method="BFGS",
                jac = self.jacobian(),
                # method="cobyla",
                # method="SLSQP",
                tol=1e-3,
                # options={"maxiter":self.config['maxiter'], "rhobeg":0.1},
                options = {
                'rhobeg': step_size,
                'ftol': 2.220446049250313e-09,  # Function tolerance
                'gtol': 1e-08,  # Gradient tolerance
                # 'eps': 1e-08,  # Step size for numerical approximation
                'maxiter': self.config['maxiter'],  # Maximum iterations
                'maxfun': self.config['maxiter'],  # Maximum function evaluations
                },
                # options={"maxiter":self.config['maxiter']},
                # bounds= [(0, 2*np.pi) for _ in range(self.ansatz.num_parameters)]
            )

        print(res)
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
    jax.debug.print("Output value: {y}", y=prob)
    return prob


def my_optimizer(x, cost_func, eps, step_size, approx_min, log):
    M = 200
    l = 1
    s = 1
    c = lambda s: jnp.exp((M - s) / l)
    loss = lambda x,s: 1 * jnp.exp(-(c(s)  /  (np.exp( 1e2 * (cost_func(x) - approx_min)) - 1)**2 )**2)
    grad_loss = lambda x,s: approx_fprime(x, loss, eps, s)
    tmp_grad  = grad_loss(x,s)

    current_cost = cost_func(x) 
    current_loss = loss(x,s)
    while(np.abs(current_cost - approx_min - 1e-5) > 1e-4):
        while(current_loss < 0.4):
            s+= 1
            current_loss = loss(x,s)
        
        grad  = grad_loss(x,s)
        grad_overlap = jnp.dot(normalize(tmp_grad),normalize(grad))
        if grad_overlap > 0.995 and step_size < 0.01:
            step_size *= 1.05
        if grad_overlap < 0:
            grad = tmp_grad
            x = x + normalize(grad) * step_size * 0.5
            step_size *= 0.95
            continue

        tmp_grad = grad
        x = x - normalize(grad) * step_size 
        current_cost, energy, overlap= cost_func(x, return_all=True)
        current_loss = loss(x,s)
        log_dict = {"cost": current_cost, "lost": current_loss, "grad_overlap":grad_overlap, "step_size":step_size, "energy": energy, "overlap": overlap, "s": s}
        if log:
            wandb.log(log_dict)
        else:
            print(log_dict)
    
    print(f"Optimization Done!\n function min = {current_cost}")
    res = Optimizer_reuslt(function_value = current_cost,x = x)
    return res

def my_optimizer_V2(x, cost_func, eps, step_size, approx_min, log):
    M = 4
    s = 1
    c = lambda s: jnp.exp((M - s) / 1)
    loss = lambda x,s: jnp.exp( - (c(s) / ( (np.exp( 1e1 * (cost_func(x) - approx_min - 1) ) ) - 1)**2)**0.5)
    # loss = lambda x,s: -np.exp( -10 * (cost_func(x) - approx_min - 1))
    grad_loss = lambda x,s: approx_fprime(x, loss, eps, s)
    tmp_grad  = grad_loss(x,s)
    current_cost = cost_func(x) 
    current_loss = loss(x,s)
    grad_step_size = 1e-2
    counter = 0
    while(np.abs(current_cost - approx_min) > 1e-4):
        # print("\n\nLowering bounds")
        # while(current_loss < 0.5):
        #     s+= 1
        #     current_loss = loss(x,s)
    
        grad_overlap_counter = 0
        direction_vector = jnp.zeros(len(x))
        print("Finding grad")
        while(grad_overlap_counter < 5):
    
            grad  = grad_loss(x,s)
            grad_overlap = jnp.dot(normalize(tmp_grad),normalize(grad))
            tmp_grad = grad
            x = x - normalize(grad) * grad_step_size
            current_cost, energy, overlap= cost_func(x, return_all=True)
            current_loss = loss(x,s)
            log_dict = {"cost": current_cost, "lost": current_loss, "grad_overlap":grad_overlap, "step_size":step_size, "energy": energy, "overlap": overlap, "s": s}
            if log:
                wandb.log(log_dict)
            else:
                print(log_dict)
            if jnp.abs(current_cost - approx_min) < 1e-4:
                break
            if grad_overlap > 0.995:
                grad_overlap_counter += 1
                direction_vector += grad
            else:
                grad_overlap_counter = 0
                direction_vector = jnp.zeros(len(x))
                grad_step_size /= 2
                
        
        direction_vector = normalize(direction_vector)
        # print("Tunning learning rate")
        # while(current_loss  >= loss(x - direction_vector * step_size,s) and step_size <= 2 * jnp.pi):
        #     step_size *= 2
        #     print(f"Learing rate:{step_size}, curret_loss: {current_loss}, loss: {loss(x - direction_vector * step_size,s)}")
        # while(current_loss * 2 <= loss(x - direction_vector * step_size,s)):
        #     step_size /= 1.1
        #     print(f"Learing rate:{step_size}, curret_loss: {current_loss}, loss: {loss(x - direction_vector * step_size,s)}")
        # x = x - direction_vector * step_size / 2 
        
        loss_value_list = []
        cost_value_list = []
        l_range = jnp.linspace(start = - 5 * step_size, stop =   5 * step_size, num = int(1e2))
        # l_range = jnp.linspace(start = -2 * jnp.pi, stop =   2 * jnp.pi, num = int(5e2))
        for l in tqdm(l_range):
            loss_value_list.append(loss(x - direction_vector * l,s))
            cost_value_list.append(cost_func(x - direction_vector * l))
        plt.figure()
        plt.plot(l_range,loss_value_list)
        plt.plot(l_range,cost_value_list)
        plt.grid()
        plt.savefig(f'tmpfig_{counter}.jpg')
        plt.close()
        counter += 1
        
        
        print("Optimize")
        res = minimize(
            # lambda l: cost_func(x - direction_vector * step_size * l, return_all=False),
            lambda l: loss(x - direction_vector * step_size * l,s),
            0,
            args=(),
            # method="cobyla",
            method="SLSQP",
            # tol=1e-5,
            # options={"maxiter":100, "rhobeg":step_size / 10},
        )
        x = x - direction_vector * step_size * res.x
        current_cost, energy, overlap= cost_func(x, return_all=True)
        current_loss = loss(x,s)
        log_dict = {"cost": current_cost, "lost": current_loss, "grad_overlap":grad_overlap, "step_size":step_size, "energy": energy, "overlap": overlap, "s": s}
        log_dict = {"cost": current_cost, "lost": current_loss, "grad_overlap":grad_overlap, "step_size":step_size, "energy": energy, "overlap": overlap, "s": s}
        if log:
            wandb.log(log_dict)
        else:
            print(log_dict)
  

    print(f"Optimization Done!\n function min = {current_cost}")
    res = Optimizer_reuslt(function_value = current_cost,x = x)
    return res