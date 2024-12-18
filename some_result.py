#%%
from jax_simulation import *
import pickle

#%%
# Comparing the phases of the magnetic IQH and the FQH
Nx = 2
Ny = 6
cite_number = 2 * Nx * Ny
n = cite_number // 6

print("---1---")
try:
    IQH_state = np.load(f'data/Nx-{Nx}_Ny-{Ny}_q=3_magnetic.npy')
    mps = Multi_particle_state(N = 2 * Nx * Ny,n=n)
except:
    print("Calculting IQH state")
    IQH_state, mps = create_IQH_in_extendend_lattice(Nx,Ny,n,extention_factor = 1, band_energy = 1, H_sb = H_real_space_magnetic)
    np.save(f'data/Nx-{Nx}_Ny-{Ny}_q=3_magnetic',IQH_state)

print("---2----")
try:
    loaded = np.load(f'data/Nx-{Nx}_Ny-{Ny}_k-4.npz')
    eigenvectors = loaded['a']
except:
    print("Calculting FQH states")
    eigenvalues, eigenvectors = exact_diagnolization(Nx, Ny, n=n, band_energy=band_energy, interaction_strength=interaction_strength,k=4,multi_process=False, save_result=False, show_result=False)


for i, FQH in enumerate(eigenvectors.T):
    # print_mp_state(FQH,Nx,Ny,mps)
    delta_phase = np.angle(IQH_state / FQH)
    plt.figure()
    plt.hist(delta_phase,bins=100)
    plt.title(f"arg(IQH_magnetic / FQH_{i + 1})")
    plt.figure()
    plt.scatter(np.angle(IQH_state), np.angle(FQH), color='blue', marker='.')

# Customize the plot
    plt.title(f"Scatter Plot for phases {i + 1}")
    plt.xlabel("arg(IQH_magnetic)")
    plt.ylabel(f"arg(FQH_{i + 1})")

    # Show the plot
    plt.show()

#%%
# Comparison between estimiated phase from VQE to theta function
Nx = 2
Ny = 6
cite_number = 2 * Nx * Ny
n = cite_number // 6
mps = Multi_particle_state(N = 2 * Nx * Ny,n=n)

ansatz = Ansatz_class(Nx,Ny,mps)

navie_theta_guess = ansatz.flux_get_inital_params()

# Load data from pickle file
path = '/Users/dor/Documents/technion/Master/research/code/RFCI/results/vqe_simulation/jax/Nx-2_Ny-6_p--1_q-3/ixq16ntm/optimization_0/res.pickle'
with open(path, 'rb') as file:
    loaded_data = pickle.load(file)
vqe_values = loaded_data.x

plt.figure()
plt.scatter(navie_theta_guess, vqe_values, color='blue', marker='.')

# Customize the plot
plt.title(f"Scatter Plot for estimiated phases structre")
plt.xlabel("theta(zi-zj)")
plt.ylabel(r"VQE f(ij)")

# Show the plot
plt.show()