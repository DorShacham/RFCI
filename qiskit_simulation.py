#%%
import qiskit_nature, qiskit
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit import QuantumRegister,ClassicalRegister
from qiskit.quantum_info.operators import Operator
from qiskit.visualization import array_to_latex
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

from IQH_state import *
from flux_attch import *


#%%
# Initialzing state
# Preparing the Full Hilbert 2^N state
Nx = 2
Ny = 2
# number of electrons - half of the original system
n = Nx * Ny 
extention_factor = 3

state, mps = create_IQH_in_extendend_lattice(Nx = Nx, Ny = Ny, extention_factor = extention_factor)

Nx = extention_factor * Nx
N = 2 * Nx * Ny

state_vector = state_2_full_state_vector(state, mps)

#%%
# Build the phase attechment operator

# build a 2 qubit gate adding the phase @angle if both electron in the i,j sites
# return the gate
def uij(i,j):
    unitary = np.eye(4, dtype = complex)
    za = cite_index_2_z(i, mps, Ny)
    zb = cite_index_2_z(j, mps, Ny)
    unitary[-1,-1] = np.exp(2j * np.angle(za - zb))
    Op = Operator(unitary).to_instruction()
    Op.label = str(f"{i},{j}")
    return Op

qc = QuantumCircuit(N)
qc.initialize(state_vector, range(N))

for i in range(N):
    for j in range(i + 1,N):
        u = uij(i,j)
        qc.append(u,[i,j])

# qc.draw('mpl')

#%% 
# Run circit

simulator = Aer.get_backend('statevector_simulator')
# Run and get the result object
result = simulator.run(qc).result()
new_state_vector = np.array(result.get_statevector())

#%% test
test_state = flux_attch_2_compact_state(state, mps, Ny)
test_state = state_2_full_state_vector(test_state, mps)

print(np.linalg.norm(new_state_vector - test_state))
print(np.linalg.norm(new_state_vector + test_state))
# %%


# %%

# Initialzing state
# Preparing the Full Hilbert 2^N state
Nx = 2
Ny = 2
# number of electrons - half of the original system
n = Nx * Ny 
extention_factor = 3

state, mps = create_IQH_in_extendend_lattice(Nx = Nx, Ny = Ny, extention_factor = extention_factor)

Nx = extention_factor * Nx
N = 2 * Nx * Ny

state_vector = state_2_full_state_vector(state, mps)

H = build_H(Nx = Nx, Ny = Ny)
hamiltonian_terms = {}
for i in range(N):
    for j in range(N):
        hamiltonian_terms[f"+_{i} -_{j}"] = H[i,j]

hamiltonian_terms = FermionicOp(hamiltonian_terms, num_spin_orbitals=N)
qubit_converter = JordanWignerMapper()
qiskit_H = qubit_converter.map(hamiltonian_terms)

from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

# Create your quantum circuit
qc = QuantumCircuit(N)
qc.initialize(state_vector, range(N))



from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator, Statevector
from qiskit.opflow import StateFn, PauliExpectation, CircuitStateFn

backend = Aer.get_backend('statevector_simulator')
job = execute(qc, backend)
statevector = job.result().get_statevector()

# Create your operator (for example, a Pauli Z on the first qubit)

# Calculate the expectation value
expectation_value = StateFn(qiskit_H, is_measurement=True) @ StateFn(statevector)
print(f"Expectation value: {expectation_value.eval().real}")