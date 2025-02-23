#%%
import os
import platform
import multiprocessing


# Check the operating system
if platform.system() == "Linux":
    # Set environment variables to limit CPU usage on Linux
    os.environ["OMP_NUM_THREADS"] = "10"
    os.environ["MKL_NUM_THREADS"] = "10"
    os.environ["NUMEXPR_NUM_THREADS"] = "10"
    os.environ["OPENBLAS_NUM_THREADS"] = "10"
    os.environ["JAX_NUM_THREADS"] = "10"
    print("CPU usage limited to 10 threads on Linux.")
elif platform.system() == "Darwin":
    # macOS-specific behavior (no limitation)
    print("Running on macOS. No CPU limitation applied.")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())
else:
    print("Operating system not recognized. No changes applied.")



import jax
import yaml
from exact_diagnolization import *
import jax_simulation
import wandb

def lambda_constructor(loader, node):
    return eval(f"lambda {node.value}")

yaml.add_constructor('!lambda', lambda_constructor, Loader=yaml.FullLoader)

config_file = './configs/jax/config_for_expressiv_test.yaml'
with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
config_list_keys = list(config)
config_list = [config[key] for key in config_list_keys if key is not None and 'config' in key]

Nx = config['data']['Nx']
Ny = config['data']['Ny']
n = config['data']['n']
p = config['data']['p']
q = config['data']['q']

overlap_array = []
for layer_numer in range(10):
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
    saving_path = str(f"results/vqe_simulation/jax/Nx-{Nx}_Ny-{Ny}_p-{p}_q-{q}/{id}")
    config_list[0]['flux_gate_true'] = True
    config_list[0]['layer_numer'] = layer_numer
    res = jax_simulation.vqe_simulation(Nx = Nx, Ny = Ny, config_list = config_list, n = n, p=p, q=q, pre_ansatz = None,saveto = saving_path, log = log)
