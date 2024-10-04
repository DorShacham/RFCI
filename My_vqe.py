#%%
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator, QasmSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import Statevector


# VQE impliminatation: taking @ansatz @hamiltonian and find paramters 
# for the @ansatz that minimizes the @hamiltonian expection value according to my_estimator.
# @initial_state is the state of the quantum ciricut before the anzats
class VQE:
    def __init__(self, initial_state ,ansatz,hamiltonian, maxiter = 1e5):
        self.initial_state = initial_state
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.cost_history_dict = {
            "prev_vector": None,
            "iters": 0,
            "cost_history": [],
        }
        self.maxiter = maxiter
        self.res = None

# calculate the cost_function - the expection value of self.hamiltonian according to self.estimator
# with self.anzats(@params)
    def cost_func(self,params):
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
        # pub = (self.ansatz, [self.hamiltonian], [params])
        # result = self.estimator.run(pubs=[pub]).result()
        # energy = result[0].data.evs[0]
        energy = my_estimator(self.initial_state, self.ansatz.assign_parameters(params), self.hamiltonian).real

        self.cost_history_dict["iters"] += 1
        self.cost_history_dict["prev_vector"] = params
        self.cost_history_dict["cost_history"].append(energy)
        print(f"Iters. done: {self.cost_history_dict['iters']} [Current cost: {energy}]")

        return energy

# start the optimization proccess. all data on optimization is saved in self.cost_history_dict
    def minimize(self):

        # x0 = 2 * np.pi * np.random.random(self.ansatz.num_parameters)
        x0 = np.zeros(self.ansatz.num_parameters)

        res = minimize(
            self.cost_func,
            x0,
            args=(),
            method="cobyla",
            # tol=0.01,
            options={"maxiter":self.maxiter},
        )
        print(res)
        self.res = res
        return res

# plot cost function as a function of iterations.
    def plot(self):
        plt.figure()
        plt.plot(range(self.cost_history_dict["iters"]), self.cost_history_dict["cost_history"])
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.draw()


# Calculate the expection value of @operator on final state from @qc with @initial_state
def my_estimator(initial_state,qc,operator):
    sv = Statevector(initial_state)
    sv = sv.evolve(qc)
    return sv.expectation_value(operator)