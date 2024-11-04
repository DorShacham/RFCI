#%%
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
from qiskit_algorithms.optimizers import SPSA
import matplotlib.pyplot as plt
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator, QasmSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import Statevector
import pickle
import wandb
from scipy.optimize import approx_fprime
from functools import partial
from tqdm import tqdm


from IQH_state import print_state_vector, normalize


class Optimizer_reuslt:
    def __init__(self,function_value, x):
        self.function_value = function_value
        self.x = x

# VQE impliminatation: taking @ansatz @hamiltonian and find paramters 
# for the @ansatz that minimizes the @hamiltonian expection value according to my_estimator.
# @initial_state is the state of the quantum ciricut before the ansatz
# if @saveto - is not None, save the result of the optimization and graph to this addres
class VQE:
    def __init__(self,Nx, Ny, initial_state ,ansatz,hamiltonian, config, approx_min = None, saveto = None, log = False, config_i = None, ground_states = None):
        self.initial_state = initial_state
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.cost_history_dict = {
            "prev_vector": None,
            "iters": 0,
            "cost_history": [],
        }
        self.config = config
        self.Nx = Nx
        self.Ny = Ny
        self.res = None
        self.path = saveto
        self.log = log
        self.config_i = config_i
        self.ground_states = ground_states
        self.approx_min = approx_min

        if config['loss'] is None:
            self.loss = lambda x: x
        else:
            self.loss = config['loss']

        if config['cooling_protocol']:
            self.loss = lambda x,c:  1 * np.exp(-(c  /  (np.exp( 100 * (x + 1)) - 1)**2 )**0.05)

        

# calculate the cost_function - the expection value of self.hamiltonian according to self.estimator
# with self.ansatz(@params)
    def cost_func(self,params, for_grad = False):
        """Return estimate of energy from estimator

        Parameters:
            params (ndarray): Array of ansatz parameters
            ansatz (QuantumCircuit): Parameterized ansatz circuit
            hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
            estimator (EstimatorV2): Estimator primitive instance
            cost_history_dict: Dictionary for storing intermediate results

        Returns:
            float: Energy estimate
        """
        energy = my_estimator(self.initial_state, self.ansatz.assign_parameters(params), self.hamiltonian).real
        cost = energy

        if not for_grad:
            self.cost_history_dict["iters"] += 1
            self.cost_history_dict["prev_vector"] = params
            self.cost_history_dict["cost_history"].append(energy)

            if (self.config["cktp_iters"] is not None) and  (self.cost_history_dict["iters"] % self.config["cktp_iters"] == 0) and self.log:
                print_state_vector(Statevector(self.initial_state).evolve(self.ansatz.assign_parameters(params)).data, self.Nx, self.Ny, saveto=str(self.path) + str(f'/electron_density_{self.cost_history_dict["iters"]}.jpg'))
                wandb.log({"Electron Density": wandb.Image(str(self.path) + str(f'/electron_density_{self.cost_history_dict["iters"]}.jpg'), caption=f"Config {self.config_i} iter {self.cost_history_dict['iters']}")}, commit = False) 
        
        if self.ground_states is not None:
            overlap = sbuspace_probability(Statevector(self.initial_state).evolve(self.ansatz.assign_parameters(params)), subspace = self.ground_states)
            if (not for_grad) and self.log:
                wandb.log({f'Overlap_{self.config_i}': overlap}, commit=False)
            if self.config['overlap_optimization']:
                cost  = -overlap

        if self.config['cooling_protocol']:
            s = self.cost_history_dict["iters"]
            M = 200
            l = 1
            c = np.exp((M - s) / l)
            cost = self.loss(cost , c)
        else:
            cost = self.loss(cost)
        
        if (not for_grad) and self.log:
            wandb.log(
                {
                    f'Energy_config_{self.config_i}': energy,
                    f'Cost_config_{self.config_i}': cost,
                }
            )
        elif not for_grad:
            print(f"Iters. done: {self.cost_history_dict['iters']} [Current cost: {cost}, energy:{energy}]")

        return cost

    def my_cost_func(self,params, return_all = False):
        energy = my_estimator(self.initial_state, self.ansatz.assign_parameters(params), self.hamiltonian).real
        cost = energy

        if (self.ground_states is not None) and self.config['overlap_optimization']:
            overlap = sbuspace_probability(Statevector(self.initial_state).evolve(self.ansatz.assign_parameters(params)), subspace = self.ground_states)
            cost  = -overlap
        if return_all:
            return cost, energy, overlap
        else:
            return cost
         

# start the optimization proccess. all data on optimization is saved in self.cost_history_dict
    def minimize(self):
        if self.config['random_initial_parametrs']:
            x0 = 2 * np.pi * np.random.random(self.ansatz.num_parameters) 
        else:
            x0 = np.zeros(self.ansatz.num_parameters)

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
                # method="cobyla",
                # method="SLSQP",
                # tol=0.00000001,
                # options={"maxiter":self.config['maxiter'], "rhobeg":0.1},
                options = {
                'ftol': 2.220446049250313e-09,  # Function tolerance
                'gtol': 1e-08,  # Gradient tolerance
                'eps': 1e-08,  # Step size for numerical approximation
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


# Calculate the expection value of @operator on final state from @qc with @initial_state
def my_estimator(initial_state,qc,operator):
    sv = Statevector(initial_state)
    sv = sv.evolve(qc)
    return sv.expectation_value(operator)

# Calculate the probatility of @state_vector to be in the subspace spanned by the set @subspace
def sbuspace_probability(state_vector, subspace):
    prob = 0
    for v in subspace:
        prob += np.abs(state_vector.inner(v)) ** 2
    return prob


def my_optimizer(x, cost_func, eps, step_size, approx_min, log):
    M = 200
    l = 1
    s = 1
    c = lambda s: np.exp((M - s) / l)
    loss = lambda x,s: 1 * np.exp(-(c(s)  /  (np.exp( 1e2 * (cost_func(x) - approx_min)) - 1)**2 )**2)
    grad_loss = lambda x,s: approx_fprime(x, loss, eps, s)
    tmp_grad  = grad_loss(x,s)

    current_cost = cost_func(x) 
    current_loss = loss(x,s)
    while(np.abs(current_cost - approx_min - 1e-5) > 1e-4):
        while(current_loss < 0.4):
            s+= 1
            current_loss = loss(x,s)
        
        grad  = grad_loss(x,s)
        grad_overlap = np.dot(normalize(tmp_grad),normalize(grad))
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
    c = lambda s: np.exp((M - s) / 1)
    loss = lambda x,s: np.exp( - (c(s) / ( (np.exp( 1e1 * (cost_func(x) - approx_min - 1) ) ) - 1)**2)**0.5)
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
        direction_vector = np.zeros(len(x))
        print("Finding grad")
        while(grad_overlap_counter < 5):
    
            grad  = grad_loss(x,s)
            grad_overlap = np.dot(normalize(tmp_grad),normalize(grad))
            tmp_grad = grad
            x = x - normalize(grad) * grad_step_size
            current_cost, energy, overlap= cost_func(x, return_all=True)
            current_loss = loss(x,s)
            log_dict = {"cost": current_cost, "lost": current_loss, "grad_overlap":grad_overlap, "step_size":step_size, "energy": energy, "overlap": overlap, "s": s}
            if log:
                wandb.log(log_dict)
            else:
                print(log_dict)
            if np.abs(current_cost - approx_min) < 1e-4:
                break
            if grad_overlap > 0.995:
                grad_overlap_counter += 1
                direction_vector += grad
            else:
                grad_overlap_counter = 0
                direction_vector = np.zeros(len(x))
                grad_step_size /= 2
                
        
        direction_vector = normalize(direction_vector)
        # print("Tunning learning rate")
        # while(current_loss  >= loss(x - direction_vector * step_size,s) and step_size <= 2 * np.pi):
        #     step_size *= 2
        #     print(f"Learing rate:{step_size}, curret_loss: {current_loss}, loss: {loss(x - direction_vector * step_size,s)}")
        # while(current_loss * 2 <= loss(x - direction_vector * step_size,s)):
        #     step_size /= 1.1
        #     print(f"Learing rate:{step_size}, curret_loss: {current_loss}, loss: {loss(x - direction_vector * step_size,s)}")
        # x = x - direction_vector * step_size / 2 
        
        loss_value_list = []
        cost_value_list = []
        l_range = np.linspace(start = - 5 * step_size, stop =   5 * step_size, num = int(1e2))
        # l_range = np.linspace(start = -2 * np.pi, stop =   2 * np.pi, num = int(5e2))
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