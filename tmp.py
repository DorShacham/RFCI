import qiskit_nature, qiskit
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.compiler import transpile, assemble
from qiskit.visualization import *
from qiskit import QuantumRegister,ClassicalRegister
from qiskit.quantum_info.operators import Operator
from qiskit.visualization import array_to_latex
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit.circuit import ParameterVector
from qiskit.circuit import Parameter
from qiskit.circuit.library import ExcitationPreserving
from qiskit_aer import AerSimulator, QasmSimulator
from qiskit.primitives import BackendEstimatorV2
from qiskit.primitives import StatevectorEstimator




from scipy.linalg import expm



from IQH_state import *
from flux_attch import *


#%%

# Build the Hamiltonina Operator (with interction) from sparce Pauli strin for a lattice of shape (@Nx, @Ny).
# If @return_NN = True then return a list of (n1,n2) nearest niegbors for anaztas entanglement.
def build_qiskit_H(Nx, Ny, reutrn_NN = True):
    N = 2 * Nx * Ny
    H = build_H(Nx = Nx, Ny = Ny)
    hamiltonian_terms = {}
    # single body
    for i in range(N):
        for j in range(N):
            hamiltonian_terms[f"+_{i} -_{j}"] = H[i,j]

    # interaction
    NN = []
    for x in range(Nx):
        for y in range(Ny):
            n1 = cite_2_cite_index(x = x, y = y, sublattice = 0, Ny = Ny)
            for i in [0,1]:
                for j in [0,1]:
                    n2 = cite_2_cite_index(x = (x - i) % Nx, y = (y - j) % Ny, sublattice = 1, Ny = Ny)
                    hamiltonian_terms[f"+_{n1} -_{n1} +_{n2} -_{n2}"] = 0
                    NN.append((n1,n2))
            
    hamiltonian_terms = FermionicOp(hamiltonian_terms, num_spin_orbitals=N)
    qubit_converter = JordanWignerMapper()
    qiskit_H = qubit_converter.map(hamiltonian_terms)

    if reutrn_NN:
        return qiskit_H, NN
    else:
        return qiskit_H

def vqe_ansatz(N, initail_state_vector):
    qc = QuantumCircuit(N)
    qc.initialize(initail_state_vector, range(N))



    ansatz = ExcitationPreserving(N, reps=3, insert_barriers=True, entanglement='linear')

    qc.compose(ansatz, inplace= True)
    return qc



Nx = 1
Ny = 2
# number of electrons - half of the original system
n = Nx * Ny 
extention_factor = 1

state, mps = create_IQH_in_extendend_lattice(Nx = Nx, Ny = Ny, extention_factor = extention_factor)
Nx = extention_factor * Nx
N = 2 * Nx * Ny
# state = flux_attch_2_compact_state(state, mps, Ny)
state_vector = state_2_full_state_vector(state, mps)

qiskit_H, NN = build_qiskit_H(Nx = Nx, Ny = Ny, reutrn_NN=True)
qc_inital_state = QuantumCircuit(N)
qc_inital_state.initialize(state_vector, range(N))

backend = AerSimulator(method='statevector') 
estimator = BackendEstimatorV2(backend=backend)
estimator = StatevectorEstimator()
qc = qc_inital_state
H = qiskit_H
pub = (qc_inital_state, qiskit_H)
job = estimator.run([pub])
result = job.result()[0]

print(f"Expectation value: {result.data.evs}")