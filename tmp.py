#%%
from exact_diagnolization import *
from qiskit_simulation import *

if __name__ == "__main__":
    # eigenvalues, eigenvectors = exact_diagnolization(Nx=6, Ny=3, multi_process=True, max_workers=10, multiprocess_func=multiprocess_map,from_memory=False)
    Nx = 1
    Ny = 3
    extention_factor = 3
    saving_path = str(f"results/vqe_simulation/Nx-{Nx}_Ny-{Ny}_EF-{extention_factor}")
    config1 = {"band_energy": 1e0, "interaction_strength": 0, "translation_invariant_ansatz": False, "anzts_reps": 3, "flux_attch": True, "maxiter": 9e2, "NNN": True}
    # config2 = {"band_energy": 1e0, "interaction_strength": 1e-1, "translation_invariant_ansatz": False, "anzts_reps": 3, "flux_attch": False, "maxiter": 5e2,  "NNN": False}
    # config3 = {"band_energy": 1e0, "interaction_strength": 1e-1, "translation_invariant_ansatz": False, "anzts_reps": 3, "flux_attch": False, "maxiter": 1e2,  "NNN": False}
    # config_list = [config1,config2,config3]
    config_list = [config1,config2]
    vqe_simulation(Nx = Nx, Ny = Ny, config_list = config_list, n = None, extention_factor = extention_factor , pre_anzats = None,saveto = saving_path)
    
#%%
import numpy as np
e = np.array(sorted([-5.95450488, -5.95446248, -5.95446248, -5.94610191, -5.9462678 , -5.94564128, -5.94592613, -5.94626068, -5.94626068, -5.94592613]))
print(e - np.min(e))