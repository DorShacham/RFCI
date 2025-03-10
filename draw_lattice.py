# %%

import matplotlib.pyplot as plt
import numpy as np

from IQH_state import *

def build_H(Nx = 2, Ny = 2, band_energy = 1, M = 0, phi = np.pi/4, phase_shift_x = 0, phase_shift_y = 0, element_cutoff= None):
# parametrs of the model
    N = Nx * Ny
    # phi = np.pi/4
    t1 = 1
    t2 = (2-np.sqrt(2))/2 * t1
    # t2 = t1 / np.sqrt(2)

    # Building the single particle hamiltonian (h2)
    # need to check if the gauge transformation is needed to adress (Natanel said no)
    # Starting the BZ from zero to 2pi since this is how the DFT matrix is built
    
    Kx = np.linspace(0, 2 * np.pi,num=Nx,endpoint=False)
    Ky = np.linspace(0, 2 * np.pi,num=Ny,endpoint=False)
    # Kx = np.linspace(-np.pi, np.pi,num=Nx,endpoint=False)
    # Ky = np.linspace(-np.pi, np.pi,num=Ny,endpoint=False)
    X = np.array(range(Nx)) 
    Y = np.array(range(Ny)) 

    def build_h2(kx, ky, band_energy):
        h11 = 2 * t2 * (np.cos(kx) - np.cos(ky)) + M
        h12 = t1 * np.exp(1j * phi) * (1 + np.exp(1j * (ky - kx))) + t1 * np.exp(-1j * phi) * (np.exp(1j * (ky)) + np.exp(1j * (- kx)))
        # h12 = 4 * t1 * np.cos(phi) * (np.cos(kx / 2) * np.cos(ky / 2)) - 1j * 4 * t1 * np.sin(phi) * (np.sin(kx / 2) * np.sin(ky / 2))
        # h2 = np.array([[h11, h12], [np.conjugate(h12), -h11]])
        h2 = np.array([[h11, np.conjugate(h12)], [h12, -h11]])
        return h2

    H_k_list = []
    i = 0
    for kx in Kx:
        for ky in Ky:
            H_single_particle = build_h2(kx + phase_shift_x/Nx,ky + phase_shift_y/Ny, band_energy)
            eig_val, eig_vec = np.linalg.eigh(H_single_particle)
            h_flat = H_single_particle / np.abs(eig_val[0]) * band_energy + i * 1e-8  # flat band limit + small disperssion for numerical stabilty
            # H_k_list.append(h_flat)
            H_k_list.append(H_single_particle)
            i += 1
            
    # creaing a block diagonal H_k matrix and dft to real space

    H_k = block_diag(*H_k_list)

    # dft matrix as a tensor protuct of dft in x and y axis and idenity in the sublattice
    dft_matrix = np.kron(dft(Nx, scale='sqrtn'),(np.kron(dft(Ny, scale='sqrtn'),np.eye(2))))
    # dft_matrix = np.kron(lattice_dft(Nx),(np.kron(lattice_dft(Ny),np.eye(2))))
    H_real_space = dft_matrix @ H_k @ dft_matrix.T.conjugate()
    # H_real_space = dft_matrix.T.conjugate() @ H_k @ dft_matrix

    if element_cutoff is not None:
        H_real_space[np.abs(H_real_space) < element_cutoff] = 0
    
    return H_real_space

Nx = 5
Ny = 5

H = build_H(Nx,Ny)

cx = 0.5
cy = -0.5

# Set hopping terms
lines = []
x0 = Nx//2 + 1
y0 = Ny//2 - 1
c0 = 0
for i,t in enumerate(H[:,cite_2_cite_index(x=x0,y=y0,sublattice=c0,Ny=Ny)]):
    if np.abs(t) > 1e-5:
        c = i % 2
        y = ((i - c) // 2) % Ny
        x = (((i - c) // 2) - y) // Ny
        print(f"{(x-x0,y-y0,(c - c0)%2)} : {np.abs(t),np.angle(t) / np.pi}")
        lines.append({'start': [x0 + cx * c0, y0 + cy *c0], 'end': [x + c * cx, y + c* cy], 't': t})



x0 = Nx//2
y0 = Ny//2
c0 = 0
for i,t in enumerate(H[:,cite_2_cite_index(x=x0,y=y0,sublattice=c0,Ny=Ny)]):
    if np.abs(t) > 1e-5:
        c = i % 2
        y = ((i - c) // 2) % Ny
        x = (((i - c) // 2) - y) // Ny
        print(f"{(x-x0,y-y0,(c - c0)%2)} : {np.abs(t),np.angle(t) / np.pi}")
        lines.append({'start': [x0 + cx * c0, y0 + cy *c0], 'end': [x + c * cx, y + c* cy], 't': t})




# Define the points
points = {
    'A': [[i,j] for i in range(Nx) for j in range(Ny)],
    'B': [[i + cx ,j + cy] for i in range(Nx) for j in range(Ny)]
}



# Plot the points
plt.figure(figsize=(8, 8))
for point_type, point_list in points.items():
    if point_type == 'A':
        plt.scatter(*zip(*point_list), marker='o', color='blue', label='Sublattice A')
    elif point_type == 'B':
        plt.scatter(*zip(*point_list), marker='*', color='red', label='Sublattice B')

# Plot the lines
for line in lines:
    plt.plot([line['start'][0], line['end'][0]], [line['start'][1], line['end'][1]], label=f'{np.abs(line["t"]):.2f}exp({np.angle(line["t"])/np.pi:.2f})*i*np.pi')
    # plt.plot([line['start'][0], line['end'][0]], [line['start'][1], line['end'][1]], label=f'{line["end"][0] - line["start"][0], line["end"][1] - line["start"][1]}')
    
    # Calculate the midpoint of the line
    midpoint_x = (line['start'][0] + line['end'][0]) / 2
    midpoint_y = (line['start'][1] + line['end'][1]) / 2
    # Place text at the midpoint
    plt.text(midpoint_x, midpoint_y, f'({np.abs(line["t"]):.2f}|({np.angle(line["t"])/np.pi:.2f}))', ha='center', va='center', color='black')

plt.axis('equal')
plt.show()
