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


from IQH_state import print_state_vector, normalize

# VQE impliminatation: taking @ansatz @hamiltonian and find paramters 
# for the @ansatz that minimizes the @hamiltonian expection value according to my_estimator.
# @initial_state is the state of the quantum ciricut before the anzats
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
# with self.anzats(@params)
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
            res = my_optimizer_V2(x = x0,cost_func=self.my_cost_func ,eps=1e-8, step_size=1e-3, approx_min=self.approx_min, log= self.log)
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
                'gtol': 1e-05,  # Gradient tolerance
                'eps': 1e-08,  # Step size for numerical approximation
                'maxiter': 15000,  # Maximum iterations
                'maxfun': 15000,  # Maximum function evaluations
                'maxcor': 10,  # Number of corrections used in the L-BFGS update
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
    loss = lambda x,s: 1 * np.exp(-(c(s)  /  (np.exp( 1e2 * (cost_func(x) - approx_min)) - 1)**2 )**1)
    grad_loss = lambda x,s: approx_fprime(x, loss, eps, s)
    tmp_grad  = grad_loss(x,s)

    current_cost = cost_func(x) 
    current_loss = loss(x,s)
    while(np.abs(current_cost - approx_min) > 1e-4):
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
        log_dict = {"cost": current_cost, "lost": current_loss, "grad_overlap":grad_overlap, "step_size":step_size, "energy": energy, "overlap": overlap}
        if log:
            wandb.log(log_dict)
        else:
            print(log_dict)
    
    print(f"Optimization Done!\n function min = {cost}")
    class Struct: pass
    res = Struct()
    res.x = x
    res.fun = current_cost
    return res

def my_optimizer_V2(x, cost_func, eps, step_size, approx_min, log):
    M = 200
    l = 1
    s = 1
    c = lambda s: np.exp((M - s) / l)
    loss = lambda x,s: 1 * np.exp(-(c(s)  /  (np.exp( 100 * (cost_func(x) - approx_min)) - 1)**2 )**0.05)
    grad_loss = lambda x,s: approx_fprime(x, loss, eps, s)
    tmp_grad  = grad_loss(x,s)
    current_cost = cost_func(x) 
    current_loss = loss(x,s)
    while(np.abs(current_cost - approx_min) > 1e-5):
        print("\n\nLowering bounds")
        while(current_loss < 0.4):
            s+= 1
            current_loss = loss(x,s)
    
        grad_overlap_counter = 0
        print("Finding grad")
        while(grad_overlap_counter < 10):
            grad  = grad_loss(x,s)
            grad_overlap = np.dot(normalize(tmp_grad),normalize(grad))
            tmp_grad = grad
            x = x - normalize(grad) * 1e-5 
            current_cost, energy, overlap= cost_func(x, return_all=True)
            current_loss = loss(x,s)
            log_dict = {"cost": current_cost, "lost": current_loss, "grad_overlap":grad_overlap, "step_size":step_size, "energy": energy, "overlap": overlap}
            if log:
                wandb.log(log_dict)
            else:
                print(log_dict)
        
            if grad_overlap > 0.999:
                grad_overlap_counter += 1
            else:
                grad_overlap_counter = 0
        
        prev_cost = current_cost
        step_size *= 0.9
        print("Approching")
        while(current_cost <= prev_cost):
            prev_cost = current_cost
            x = x - normalize(grad) * step_size
            current_cost, energy, overlap= cost_func(x, return_all=True)
            current_loss = loss(x,s)
            log_dict = {"cost": current_cost, "lost": current_loss, "grad_overlap":grad_overlap, "step_size":step_size, "energy": energy, "overlap": overlap}
            if log:
                wandb.log(log_dict)
            else:
                print(log_dict)

    print(f"Optimization Done!\n function min = {cost}")
    class Struct: pass
    res = Struct()
    res.x = x
    res.fun = cost
    return res