#%% 
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from scipy.linalg import block_diag,dft,expm
from tqdm import tqdm

from IQH_state import create_IQH_in_extendend_lattice, print_mp_state


#%%
Nx = 2
Ny = 2
extention_factor = 1

state, mps = create_IQH_in_extendend_lattice(Nx = Nx, Ny = Ny, extention_factor = extention_factor)
print_mp_state(state,Nx,Ny,mps)