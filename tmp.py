#%%
from qiskit.circuit.library import ExcitationPreserving
from qiskit.circuit import ParameterVector
import numpy as np

# 1. Create the ExcitationPreserving ansatz
num_qubits = 4
reps = 3
ansatz = ExcitationPreserving(
    num_qubits=num_qubits,
    reps=reps,
    entanglement='circular',
    mode='fsim'
)
print(ansatz.draw())
# 2. Analyze the parameter structure
print(f"Number of parameters: {ansatz.num_parameters}")
print(f"Parameter names: {ansatz.parameters}")

# 3. Create a custom parameter binding
# Determine the number of unique parameters per repetition
params_per_rep = ansatz.num_parameters // reps

# Create a ParameterVector for the unique parameters
unique_params = ParameterVector('θ', params_per_rep)

# 4. Bind the parameters to achieve translation invariance
param_dict = {}
for i, param in enumerate(ansatz.parameters):
    param_dict[param] = unique_params[i % params_per_rep]

# Bind the parameters
translation_invariant_ansatz = ansatz.assign_parameters(param_dict)

print(f"Number of parameters after binding: {translation_invariant_ansatz.num_parameters}")
print(f"Parameter names after binding: {translation_invariant_ansatz.parameters}")

# Visualize the circuit (optional)
# translation_invariant_ansatz.draw()