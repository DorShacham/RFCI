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
def build_qiskit_H(Nx, Ny, interaction_strength = 1e-1, band_energy = 1, reutrn_NN = True, NNN = False):
    N = 2 * Nx * Ny
    H = build_H(Nx = Nx, Ny = Ny, band_energy = band_energy)
    hamiltonian_terms = {}
    # single body
    for i in range(N):
        for j in range(N):
            hamiltonian_terms[f"+_{i} -_{j}"] = H[i,j]

    # interaction and nearest neighbors
    NN = []
    for i in [0,1]:
        for j in [0,1]:
            for x in range(Nx):
                for y in range(Ny):
                    n1 = cite_2_cite_index(x = x, y = y, sublattice = 0, Ny = Ny)
                    n2 = cite_2_cite_index(x = (x - i) % Nx, y = (y - j) % Ny, sublattice = 1, Ny = Ny)
                    hamiltonian_terms[f"+_{n1} -_{n1} +_{n2} -_{n2}"] = interaction_strength
                    NN.append((n1,n2))
    if NNN:
        for i,j in [(0,1),(0,-1),(1,0),(-1,0)]:
            for x in range(Nx):
                for y in range(Ny):
                    n1 = cite_2_cite_index(x = x, y = y, sublattice = 0, Ny = Ny)
                    n3 = cite_2_cite_index(x = x, y = y, sublattice = 1, Ny = Ny)
                    n2 = cite_2_cite_index(x = (x - i) % Nx, y = (y - j) % Ny, sublattice = 0, Ny = Ny)
                    n4 = cite_2_cite_index(x = (x - i) % Nx, y = (y - j) % Ny, sublattice = 1, Ny = Ny)
                    hamiltonian_terms[f"+_{n1} -_{n1} +_{n2} -_{n2}"] = interaction_strength
                    hamiltonian_terms[f"+_{n3} -_{n3} +_{n4} -_{n4}"] = interaction_strength
                    NN.append((n1,n2))
                    NN.append((n3,n4))


            
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

# create an excitation preserving translation_invariant_ansatz according to the symmetris of the problem
# For lattice of size @Nx @Ny with @reps
def translation_invariant_ansatz(Nx, Ny, reps):
    NN = []
    for i in [0,1]:
        for j in [0,1]:
            for x in range(Nx):
                for y in range(Ny):
                    n1 = cite_2_cite_index(x = x, y = y, sublattice = 0, Ny = Ny)
                    n2 = cite_2_cite_index(x = (x - i) % Nx, y = (y - j) % Ny, sublattice = 1, Ny = Ny)
                    NN.append((n1,n2))
    N = 2 * Nx * Ny
    num_qubits = N
    ansatz = ExcitationPreserving(
        num_qubits=num_qubits,
        reps=reps,
        entanglement=NN,
        mode='fsim',
        insert_barriers=False,
        flatten=True
    )

    old_parm_per_rep = N + len(NN) * 2
    param_per_single_qubit = 2 # 2 cites in cell
    param_per_NN = 2 * 2 # there are 4 NN, out of which there is reflection invaraince in both direction meaning only 2 uniqe directions. The other 2 is becuase the anzats use 2 params per pair.
    parm_per_rep = param_per_single_qubit + param_per_NN
    num_of_params = parm_per_rep * reps + param_per_single_qubit
    # Create a ParameterVector for the unique parameters
    unique_params = ParameterVector('θ', num_of_params)

    # 4. Bind the parameters to achieve translation invariance
    param_dict = {}
    for rep in range(reps):
        for i in range(N):
            sublattice = i % 2
            param_dict[ansatz.parameters[i + old_parm_per_rep * rep]] = unique_params[sublattice + parm_per_rep * rep]
        # for i in range(len(NN) // 4):
            # param_dict[ansatz.parameters[0 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[0 + param_per_single_qubit + parm_per_rep * rep]
            # param_dict[ansatz.parameters[1 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[1 + param_per_single_qubit + parm_per_rep * rep]
            
            # param_dict[ansatz.parameters[2 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[2 + param_per_single_qubit + parm_per_rep * rep]
            # param_dict[ansatz.parameters[3 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[3 + param_per_single_qubit + parm_per_rep * rep]
            # param_dict[ansatz.parameters[4 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[2 + param_per_single_qubit + parm_per_rep * rep]
            # param_dict[ansatz.parameters[5 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[3 + param_per_single_qubit + parm_per_rep * rep]

            # param_dict[ansatz.parameters[6 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[0 + param_per_single_qubit + parm_per_rep * rep]
            # param_dict[ansatz.parameters[7 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[1 + param_per_single_qubit + parm_per_rep * rep]
        for i in range(2 * len(NN)):
            param_dict[ansatz.parameters[i + N + old_parm_per_rep * rep]] = unique_params[(i % 2) + 2 * (((i + 1) // 2) % 2) +  param_per_single_qubit + parm_per_rep * rep]
    # final rotation
    for i in range(N):
            sublattice = i % 2
            param_dict[ansatz.parameters[i + old_parm_per_rep * reps]] = unique_params[sublattice + parm_per_rep * reps]

    # Bind the parameters
    translation_invariant_ansatz = ansatz.assign_parameters(param_dict)

    print(f"Number of parameters after binding: {translation_invariant_ansatz.num_parameters}")
    print(f"Parameter names after binding: {translation_invariant_ansatz.parameters}")
    return translation_invariant_ansatz

# for a given @state_vecotr on lattice @Nx,@Ny print a heatmap of the distribution of electrons.
# if @saveto is not None should be a path to save location for the heatmap

def vqe_simulation(Nx, Ny, config_list, n = None, extention_factor = 3 , pre_anzats = None,saveto = None, log = False):
    # Initialzing state
    # Prearing the Full Hilbert 2^N state
    # number of electrons - half of the original system
    if n is None:
        n = 2 * extention_factor * Nx * Ny // 6

    state, mps = create_IQH_in_extendend_lattice(Nx = Nx, Ny = Ny,n=n, extention_factor = extention_factor, band_energy=config_list[0]['band_energy'])

    Nx = extention_factor * Nx
    N = 2 * Nx * Ny
    init_state_vector = state_2_full_state_vector(state, mps)
    sv = Statevector(init_state_vector)
    if pre_anzats is not None:
        sv = sv.evolve(pre_anzats)
    

    for i, config_dict in enumerate(config_list):
        qiskit_H, NN = build_qiskit_H(Nx = Nx, Ny = Ny, interaction_strength = config_dict['interaction_strength'], band_energy = config_dict['band_energy'], reutrn_NN=True, NNN=config_dict['NNN'])
        if config_dict['flux_attch']:
            sv = sv.evolve(flux_attch_gate(N, mps, Nx, Ny))


        if config_dict['translation_invariant_ansatz']:
            ansatz = translation_invariant_ansatz(Nx, Ny, reps = config_dict['anzts_reps'])
        else:
            ansatz = ExcitationPreserving(N, reps= config_dict['anzts_reps'], insert_barriers=False, entanglement=NN,flatten=True, mode='fsim')
        
        if saveto is not None:
            path = str(saveto) + str(f'/optimization_{i}')
            os.makedirs(path, exist_ok=True)
        else:
            path = None
        vqe = VQE(initial_state=sv.data, Nx = Nx, Ny = Ny, ansatz=ansatz, hamiltonian=qiskit_H, config = config_dict, approx_min = -n * config_dict['band_energy'], saveto = path, log = log, config_i = i)
        res = vqe.minimize()
        vqe.plot()
        # calculting initial and final energy
        i_state = sv.data
        f_state = sv.evolve(ansatz.assign_parameters(res.x))

        initial_energy = my_estimator(i_state,QuantumCircuit(N),qiskit_H)
        finial_energy = my_estimator(f_state.data,QuantumCircuit(N),qiskit_H)
        if saveto is not None:
            print_state_vector(i_state,Nx,Ny,saveto=str(path) + str('/initial_state.jpg'))
            print_state_vector(f_state,Nx,Ny,saveto=str(path) + str('/final_state.jpg'))
            if log:
                wandb.log({"Electron Density": wandb.Image(str(path) + str('/initial_state.jpg'), caption=f"Initial state config {i}")})
                wandb.log({"Electron Density": wandb.Image(str(path) + str('/final_state.jpg'), caption=f"Final state config {i}")})

            with open(path + str('/data.txt'), 'w') as file:
                file.write(str(config_dict))
                file.write(f"\ninitial_energy = {initial_energy.real}")
                file.write(f"\nfinial_energy = {finial_energy.real}")
                file.write(f"\noptimization solution = {res}")
        else:
            print_state_vector(i_state,Nx,Ny)
            print_state_vector(f_state,Nx,Ny)
            print(f"initial_energy = {initial_energy}")
            print(f"finial_energy = {finial_energy}")
            print(f"optimization solution = {res}")

        sv = f_state
