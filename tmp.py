#%%
from qiskit.circuit.library import ExcitationPreserving
from qiskit.circuit import ParameterVector
import numpy as np
from IPython.display import Image,display

from flux_attch import *

# translate the cite @index = 2 * (Ny * x + y) + sublattice -> z = x + iy
def cite_index_2_z(index,mps, Ny):
    sublattice = index % 2 
    y = (index // 2 ) % Ny
    x = index // (2 * Ny)
### maybe should use also the subllatice index

Nx = 2
Ny = 2
N = 2 * Nx * Ny

NN = []
for x in range(Nx):
    for y in range(Ny):
        n1 = cite_2_cite_index(x = x, y = y, sublattice = 0, Ny = Ny)
        for i in [0,1]:
            for j in [0,1]:
                n2 = cite_2_cite_index(x = (x - i) % Nx, y = (y - j) % Ny, sublattice = 1, Ny = Ny)
                NN.append((n1,n2))

# 1. Create the ExcitationPreserving ansatz
num_qubits = N
reps = 2
ansatz = ExcitationPreserving(
    num_qubits=num_qubits,
    reps=reps,
    entanglement=NN,
    mode='fsim',
    insert_barriers=True,
    flatten=False
)
display(ansatz.decompose().draw('mpl'))


# 2. Analyze the parameter structure
print(f"Number of parameters: {ansatz.num_parameters}")
print(f"Parameter names: {ansatz.parameters}")
#%%
# 3. Create a custom parameter binding
# Determine the number of unique parameters per repetition
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
    for i in range(len(NN) // 4):
        param_dict[ansatz.parameters[0 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[0 + param_per_single_qubit + parm_per_rep * rep]
        param_dict[ansatz.parameters[1 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[1 + param_per_single_qubit + parm_per_rep * rep]
        
        param_dict[ansatz.parameters[2 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[2 + param_per_single_qubit + parm_per_rep * rep]
        param_dict[ansatz.parameters[3 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[3 + param_per_single_qubit + parm_per_rep * rep]
        param_dict[ansatz.parameters[4 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[2 + param_per_single_qubit + parm_per_rep * rep]
        param_dict[ansatz.parameters[5 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[3 + param_per_single_qubit + parm_per_rep * rep]

        param_dict[ansatz.parameters[6 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[0 + param_per_single_qubit + parm_per_rep * rep]
        param_dict[ansatz.parameters[7 + 8 * i + N + old_parm_per_rep * rep]] = unique_params[1 + param_per_single_qubit + parm_per_rep * rep]
# final rotation
for i in range(N):
        sublattice = i % 2
        param_dict[ansatz.parameters[i + old_parm_per_rep * reps]] = unique_params[sublattice + parm_per_rep * reps]

# Bind the parameters
translation_invariant_ansatz = ansatz.assign_parameters(param_dict)

print(f"Number of parameters after binding: {translation_invariant_ansatz.num_parameters}")
print(f"Parameter names after binding: {translation_invariant_ansatz.parameters}")

# Visualize the circuit (optional)
display(translation_invariant_ansatz.decompose().draw('mpl'))