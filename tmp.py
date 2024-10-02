#%%
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
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp

from scipy.linalg import expm

from IQH_state import *
from flux_attch import *
from My_vqe import *





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

    # interaction and nearest neighbors
    NN = []
    for x in range(Nx):
        for y in range(Ny):
            n1 = cite_2_cite_index(x = x, y = y, sublattice = 0, Ny = Ny)
            for i in [0,1]:
                for j in [0,1]:
                    n2 = cite_2_cite_index(x = (x - i) % Nx, y = (y - j) % Ny, sublattice = 1, Ny = Ny)
                    hamiltonian_terms[f"+_{n1} -_{n1} +_{n2} -_{n2}"] = 10
                    NN.append((n1,n2))
            
    hamiltonian_terms = FermionicOp(hamiltonian_terms, num_spin_orbitals=N)
    qubit_converter = JordanWignerMapper()
    qiskit_H = qubit_converter.map(hamiltonian_terms)

    if reutrn_NN:
        return qiskit_H, NN
    else:
        return qiskit_H


# Build the phase attechment operator

# build a 2 qubit gate adding the phase @angle if both electron in the i,j sites
# return the gate
def uij(i,j, mps, Ny):
    unitary = np.eye(4, dtype = complex)
    za = cite_index_2_z(i, mps, Ny)
    zb = cite_index_2_z(j, mps, Ny)
    unitary[-1,-1] = np.exp(2j * np.angle(za - zb))
    Op = Operator(unitary).to_instruction()
    Op.label = str(f"{i},{j}")
    return Op


def flux_attch_gate(N, mps, Nx, Ny):
    qc = QuantumCircuit(N)
    for i in range(N):
        for j in range(i + 1,N):
            u = uij(i,j, mps, Ny)
            qc.append(u,[i,j])
    return transpile(qc)


# %%

#%%
# Initialzing state
# Preparing the Full Hilbert 2^N state
Nx = 1
Ny = 2
# number of electrons - half of the original system
n = Nx * Ny 
extention_factor = 3
Nx = extention_factor * Nx
N = 2 * Nx * Ny
qiskit_H = build_qiskit_H(Nx = Nx, Ny = Ny, reutrn_NN=False)

from qiskit.quantum_info import SparsePauliOp

# Assuming you have a SparsePauliOp object called 'sparse_pauli_op'
sparse_matrix = qiskit_H.to_matrix(sparse=True)

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# Example: Creating a random sparse matrix


# Compute k largest eigenvalues and corresponding eigenvectors
k = 100  # Number of eigenvalues/vectors to compute
eigenvalues, eigenvectors = eigsh(sparse_matrix, k=k, which='SA')
print(sorted(eigenvalues))