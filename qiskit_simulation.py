#%%
import qiskit_nature, qiskit
from qiskit import QuantumCircuit
from qiskit.visualization import array_to_latex
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import *
from qiskit.circuit import  Parameter
from qiskit import transpile
from qiskit.quantum_info.operators import Operator


from scipy.linalg import expm
import os

from IQH_state import *
from flux_attch import *
from My_vqe import *
from exact_diagnolization import exact_diagnolization




# Build the Hamiltonina Operator (with interction) from sparce Pauli strin for a lattice of shape (@Nx, @Ny).
# If @return_NN = True then return a list of (n1,n2) nearest niegbors for anaztas entanglement.
def build_qiskit_H(Nx, Ny, interaction_strength = 1e-1, band_energy = 1):
    N = 2 * Nx * Ny
    H = build_H(Nx = Nx, Ny = Ny, band_energy = band_energy)
    hamiltonian_terms = {}
    # single body
    for i in range(N):
        for j in range(N):
            hamiltonian_terms[f"+_{i} -_{j}"] = H[i,j]

    # interaction and nearest neighbors
    for i in [0,1]:
        for j in [0,1]:
            for x in range(Nx):
                for y in range(Ny):
                    n1 = cite_2_cite_index(x = x, y = y, sublattice = 0, Ny = Ny)
                    n2 = cite_2_cite_index(x = (x - i) % Nx, y = (y - j) % Ny, sublattice = 1, Ny = Ny)
                    hamiltonian_terms[f"+_{n1} -_{n1} +_{n2} -_{n2}"] = interaction_strength



            
    hamiltonian_terms = FermionicOp(hamiltonian_terms, num_spin_orbitals=N)
    qubit_converter = JordanWignerMapper()
    qiskit_H = qubit_converter.map(hamiltonian_terms)

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

def entangle_2_cites(x_vector, y_vector, sublattice_vector, Nx, Ny, param_name):
    NN = []
    qc = QuantumCircuit(2 * Nx * Ny)
    p = Parameter(str(param_name))
    for x in range(Nx):
        for y in range(Ny):
            cite_A = cite_2_cite_index(x=x,y=y,sublattice=0,Ny=Ny)
            cite_B = cite_2_cite_index(x=(x + x_vector) % Nx,y=(y + y_vector) % Ny,sublattice=(sublattice_vector) % 2, Ny=Ny)
            NN.append((cite_A,cite_B))
            qc.rxx(p,cite_A,cite_B)
    return NN, qc

# create an excitation preserving translation_invariant_ansatz according to the symmetris of the problem
# For lattice of size @Nx @Ny with @reps
def translation_invariant_ansatz(Nx, Ny, reps = 1, return_NN = False):
    N = 2 * Nx * Ny 
    band_energy = 1
    if return_NN:
        reps = 1

    ansatz = QuantumCircuit(N)
    param_counter = 0
    for res in range(reps):   
        # Rz gates
        p0 = Parameter(str(param_counter))
        p1 = Parameter(str(param_counter + 1))
        for q in range(N):
            p = p0 if (q % 2) == 0 else p1
            ansatz.rz(p,[q])
        param_counter += 2

        # NN
        NN = []
        for x_vector, y_vector in [(0,0), (0,1), (-1,0), (-1,1)]:
            NN_tmp, ansatz_tmp = entangle_2_cites(x_vector=x_vector,y_vector=y_vector,sublattice_vector=1,Nx=Nx,Ny=Ny, param_name = str(param_counter))
            NN += NN_tmp
            ansatz = ansatz.compose(ansatz_tmp, range(N))
            param_counter += 1



        # Next nearest neighbors (NNN)
        p = Parameter(str(param_counter))
        for x in range(Nx // 2):
            for y in range(Ny):
                for sublattice in [0,1]:
                    cite_A = cite_2_cite_index(x= (2 * x),y=y,sublattice=sublattice,Ny=Ny)
                    cite_B = cite_2_cite_index(x=(2 * x + 1) % Nx,y=y,sublattice=sublattice, Ny=Ny)
                    NN.append((cite_A,cite_B))
                    ansatz.rzz(p,cite_A,cite_B)

        param_counter += 1
        p = Parameter(str(param_counter))
        for x in range(Nx // 2):
            for y in range(Ny):
                for sublattice in [0,1]:
                    cite_A = cite_2_cite_index(x= (2 * x + 1),y=y,sublattice=sublattice,Ny=Ny)
                    cite_B = cite_2_cite_index(x=(2 * x + 2) % Nx,y=y,sublattice=sublattice, Ny=Ny)
                    NN.append((cite_A,cite_B))
                    ansatz.rzz(p,cite_A,cite_B)
        
        param_counter += 1
        p = Parameter(str(param_counter))
        for x in range(Nx):
            for y in range(Ny // 2):
                for sublattice in [0,1]:
                    cite_A = cite_2_cite_index(x=x,y=(2 * y),sublattice=sublattice,Ny=Ny)
                    cite_B = cite_2_cite_index(x=x,y=(2 * y + 1) % Ny,sublattice=sublattice, Ny=Ny)
                    NN.append((cite_A,cite_B))
                    ansatz.rzz(p,cite_A,cite_B)
        
        param_counter += 1
        p = Parameter(str(param_counter))
        for x in range(Nx):
            for y in range(Ny // 2):
                for sublattice in [0,1]:
                    cite_A = cite_2_cite_index(x=x,y=(2 * y + 1),sublattice=sublattice,Ny=Ny)
                    cite_B = cite_2_cite_index(x=x,y=(2 * y + 2) % Ny,sublattice=sublattice, Ny=Ny)
                    NN.append((cite_A,cite_B))
                    ansatz.rzz(p,cite_A,cite_B)
        param_counter += 1

    # final rotation - Rz gates
    p0 = Parameter(str(param_counter))
    p1 = Parameter(str(param_counter + 1))
    for q in range(N):
        p = p0 if (q % 2) == 0 else p1
        ansatz.rz(p,[q])
    param_counter += 2

    if return_NN:
        return NN
    return transpile(ansatz, optimization_level=3)


# for a given @state_vecotr on lattice @Nx,@Ny print a heatmap of the distribution of electrons.
# if @saveto is not None should be a path to save location for the heatmap

def vqe_simulation(Nx, Ny, config_list, n = None, extention_factor = 3 , pre_ansatz = None,saveto = None, log = False):
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

    if pre_ansatz is not None:
        sv = sv.evolve(pre_ansatz)
    

    for i, config_dict in enumerate(config_list):
        qiskit_H = build_qiskit_H(Nx = Nx, Ny = Ny, interaction_strength = config_dict['interaction_strength'], band_energy = config_dict['band_energy'])
        if config_dict['flux_attch']:
            sv = sv.evolve(flux_attch_gate(N, mps, Nx, Ny))


        if config_dict['translation_invariant_ansatz']:
            ansatz = translation_invariant_ansatz(Nx, Ny, reps = config_dict['anzts_reps'])
        else:
            NN = translation_invariant_ansatz(Nx, Ny, return_NN = True)
            ansatz = ExcitationPreserving(N, reps= config_dict['anzts_reps'], insert_barriers=False, entanglement=NN,flatten=True, mode='fsim')
        
        if saveto is not None:
            path = str(saveto) + str(f'/optimization_{i}')
            os.makedirs(path, exist_ok=True)
        else:
            path = None

        if config_dict['ground_state_degeneracy'] is not None:
            eigenvalues, eigenvectors = exact_diagnolization(Nx, Ny, band_energy=config_dict['band_energy'], interaction_strength=config_dict['interaction_strength'],k=config_dict['ground_state_degeneracy'],multi_process=False, save_result=False, show_result=False)
            eigenvectors = [state_2_full_state_vector(v, mps) for v in eigenvectors.T]
        else:
            eigenvectors = None

        if not config_dict['overlap_optimization']:
            approx_min = -n * config_dict['band_energy']
        else:
            approx_min = -1
        vqe = VQE(initial_state=sv.data, Nx = Nx, Ny = Ny, ansatz=ansatz, hamiltonian=qiskit_H, config = config_dict, approx_min =approx_min, saveto = path, log = log, config_i = i, ground_states = eigenvectors)
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
