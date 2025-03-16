#%%
import os
import platform
import multiprocessing



from argparse import ArgumentParser
import yaml
from exact_diagnolization import *

if __name__ == "__main__":
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('-Nx', type=int, help='Nx dimension of the lattice')
    parser.add_argument('-Ny', type=int, help='Ny dimension of the lattice')
    parser.add_argument('-cpu', type=int, help='number of cpus for multiprocess computation, if missing computes without multiprocess')

    
    args = parser.parse_args()
    
    Nx = args.Nx
    Ny = args.Ny
    cpu = args.cpu

    if cpu is None:
        cpu = 1

        # Check the operating system
    if platform.system() == "Linux":
        # Set environment variables to limit CPU usage on Linux
        os.environ["OMP_NUM_THREADS"] = str(cpu)
        os.environ["MKL_NUM_THREADS"] = str(cpu)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu)
        os.environ["JAX_NUM_THREADS"] = str(cpu)
        print(f"CPU usage limited to {cpu} threads on Linux.")
    elif platform.system() == "Darwin":
        # macOS-specific behavior (no limitation)
        print("Running on macOS. No CPU limitation applied.")
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())
    else:
        print("Operating system not recognized. No changes applied.")



import jax
from jax import config
# Enable 64-bit computation
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import yaml
from exact_diagnolization import *
import jax_simulation
import wandb

def lambda_constructor(loader, node):
    return eval(f"lambda {node.value}")

yaml.add_constructor('!lambda', lambda_constructor, Loader=yaml.FullLoader)

config_file = str(f'./configs/jax/config_for_expressiv_test_Nx-{Nx}_Ny-{Ny}.yaml')
with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
config_list_keys = list(config)
config_list = [config[key] for key in config_list_keys if key is not None and 'config' in key]

Nx = config['data']['Nx']
Ny = config['data']['Ny']
n = config['data']['n']
p = config['data']['p']
q = config['data']['q']
saving_dir = str(f"results/vqe_simulation/jax/ansatz_expressive_test/Nx-{Nx}_Ny-{Ny}_p-{p}_q-{q}")
layer_numer_range = range(1,4)


overlap_array_no_flux = []
for layer_numer in layer_numer_range:
    print(f"NO flux gate, layer number:{layer_numer}")

    id = wandb.util.generate_id()
    config['data']['id'] = id

    wandb.init(
        # set the wandb project where this run will be logged
        project="RFCI-vqe",
        name= f"Nx-{Nx}_Ny-{Ny}_p-{p}_q-{q}/expressive_test/{id}",
        id = id,
        # track hyperparameters and run metadata
        config= config
        )
    saving_path = saving_dir +  str(f"/ln-{layer_numer}_no_flux")
    config_list[0]['flux_gate_true'] = False
    config_list[0]['layer_numer'] = layer_numer
    res = jax_simulation.vqe_simulation(Nx = Nx, Ny = Ny, config_list = config_list, n = n, p=p, q=q, pre_ansatz = None,saveto = saving_path, log = True)
    overlap_array_no_flux.append(np.abs(res.fun))

overlap_array = []
for layer_numer in layer_numer_range:
    print(f"flux gate, layer number:{layer_numer}")

    id = wandb.util.generate_id()
    config['data']['id'] = id

    wandb.init(
        # set the wandb project where this run will be logged
        project="RFCI-vqe",
        name= f"Nx-{Nx}_Ny-{Ny}_p-{p}_q-{q}/expressive_test/{id}",
        id = id,
        # track hyperparameters and run metadata
        config= config
        )
    saving_path = saving_dir +  str(f"/ln-{layer_numer}_flux")
    config_list[0]['flux_gate_true'] = True
    config_list[0]['layer_numer'] = layer_numer
    res = jax_simulation.vqe_simulation(Nx = Nx, Ny = Ny, config_list = config_list, n = n, p=p, q=q, pre_ansatz = None,saveto = saving_path, log = True)
    overlap_array.append(np.abs(res.fun))



plt.figure()
plt.plot(layer_numer_range, overlap_array, ".", label = 'with flux gate')
plt.plot(layer_numer_range, overlap_array_no_flux, "x", label = 'without flux gate')
plt.xlabel("Local layers / 2")
plt.ylabel("Overlap squared in the end of optimization")
plt.grid()
plt.legend()
plt.plot()
plt.savefig(saving_dir + "/overlap_plot.jpg")