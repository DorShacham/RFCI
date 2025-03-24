#%%
from IQH_state import *


def flux_inseration_x(H_real_space, flux, Nx, Ny):
# adding adding flux with twisted boundry condition in the x direction.
    # sublattice location is (0.5,-0.5)
    cites_per_uc = 2
    x_pos = lambda x, sublattice: 1 * (x + sublattice / 2) 
    y_pos = lambda y, sublattice: 1 * (y - sublattice / 2)
    
    H_real_space_magnetic = np.array(H_real_space)
    for x1 in range(Nx):
        for y1 in range(Ny):
            for sublattice1 in range(cites_per_uc):
                cite_A_index = cite_2_cite_index(x=x1,y=y1, sublattice=sublattice1,Ny=Ny)
                for cite_B_index,t in enumerate(H_real_space[:,cite_A_index]):
                    x2, y2, sublattice2 = cite_index_2_cite(cite_B_index, Ny)
                    # mean_y =   ( y_pos(y2,sublattice2) + y_pos(y1,sublattice1) ) / 2

                    delta_x_array = np.array([x_pos(x2 ,sublattice2), x_pos(x2 + Nx ,sublattice2), x_pos(x2 - Nx ,sublattice2)]) - x_pos(x1 , sublattice1) 
                    delta_x =   np.min(np.abs(delta_x_array)) * np.sign(delta_x_array[np.argmin(np.abs(delta_x_array))])
                    if not np.array_equal(np.sort(np.abs(delta_x_array)), np.sort(np.unique(np.abs(delta_x_array)))) and np.abs(delta_x) >1e-6:
                    #     # in case of a symmetry
                        delta_x = 0
                    # delta_x = x_pos(x2 ,sublattice2) - x_pos(x1 , sublattice1) 

                    hopping_phase = np.exp(1j * 1 / Nx *  flux * delta_x )
                    H_real_space_magnetic[cite_B_index,cite_A_index] *= hopping_phase
    return H_real_space_magnetic


Nx = 12
Ny = 12

phase_shift_x = 2 * np.pi * 0.5
phase_shift_y = 0

# using k-space
H1 = build_H(Nx = Nx, Ny = Ny, phase_shift_x= phase_shift_x, phase_shift_y=phase_shift_y, flat_band=False)


# adding flux in real space
H = build_H(Nx = Nx, Ny = Ny, flat_band=False)
H2 = flux_inseration_x(H, phase_shift_x, Nx, Ny)

print(np.linalg.norm(H2 - H2.T.conjugate()))
print(np.linalg.norm(H2 - H1))
# %%

# %%

# %%
