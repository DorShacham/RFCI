#%%
from IQH_state import *
from flux_attch import *
from exact_diagnolization import *
from tqdm import tqdm

phi_list = np.linspace(start=0,stop=3, num=30)
eigenvalues_list = []
for phi in phi_list:
    eigenvalues, eigenvectors = exact_diagnolization(Nx=3, Ny=4,phase_shift_y=phi * 2 * np.pi, k=10, multi_process=False,multiprocess_func=multiprocess_map, save_result= False, show_result=False)
    eigenvalues_list.append(eigenvalues)

eigenvalues_list = np.array(eigenvalues_list) 
eigenvalues_list = eigenvalues_list - np.min(eigenvalues_list)
plt.figure()
plt.plot(phi_list,eigenvalues_list[:,0], "-.")
plt.plot(phi_list,eigenvalues_list[:,1], "-.")
plt.plot(phi_list,eigenvalues_list[:,2], "-.")
plt.plot(phi_list,eigenvalues_list[:,3], "-.")
plt.plot(phi_list,eigenvalues_list[:,4], "-.")
plt.plot(phi_list,eigenvalues_list[:,5], "-.")
plt.plot(phi_list,eigenvalues_list[:,6], "-.")
plt.grid()

# %%

#%%
plt.figure()
plt.plot(phi_list,eigenvalues_list[:,0], "-.")
plt.plot(phi_list,eigenvalues_list[:,1], "-.")
plt.plot(phi_list,eigenvalues_list[:,2], "-.")
plt.grid()
#%%
eigenvalues, eigenvectors = exact_diagnolization(Nx=3, Ny=3,n=6,interaction_strength=0, k=10, multi_process=True, save_result=False)