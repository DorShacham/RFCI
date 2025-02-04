#%%
import os
import platform
import multiprocessing


# Check the operating system
if platform.system() == "Linux":
    # Set environment variables to limit CPU usage on Linux
    os.environ["OMP_NUM_THREADS"] = "N"
    os.environ["MKL_NUM_THREADS"] = "N"
    os.environ["NUMEXPR_NUM_THREADS"] = "N"
    os.environ["OPENBLAS_NUM_THREADS"] = "N"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "N"
    os.environ["JAX_NUM_THREADS"] = "N"
    print("CPU usage limited to N threads on Linux.")
elif platform.system() == "Darwin":
    # macOS-specific behavior (no limitation)
    print("Running on macOS. No CPU limitation applied.")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())
else:
    print("Operating system not recognized. No changes applied.")



import jax

from argparse import ArgumentParser
import yaml
from exact_diagnolization import *
import qiskit_simulation
import jax_simulation
import wandb

def lambda_constructor(loader, node):
    return eval(f"lambda {node.value}")

yaml.add_constructor('!lambda', lambda_constructor, Loader=yaml.FullLoader)


if __name__ == "__main__":
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--save', action='store_true', help='Save results localy')
    
    args = parser.parse_args()
    
    config_file = args.config_path
    # config_file = './config.yaml'
    with open(config_file, 'r') as stream:
            config = yaml.load(stream, yaml.FullLoader)
    config_list_keys = list(config)
    config_list = [config[key] for key in config_list_keys if key is not None and 'config' in key]
    log = args.log

    Nx = config['data']['Nx']
    Ny = config['data']['Ny']
    n = config['data']['n']

    if config['simulation_type'] == 'jax':
        p = config['data']['p']
        q = config['data']['q']
        if log:
                id = wandb.util.generate_id()
                config['data']['id'] = id

                wandb.init(
                    # set the wandb project where this run will be logged
                    project="RFCI-vqe",
                    name= f"Nx-{Nx}_Ny-{Ny}_p-{p}_q-{q}/{id}",
                    id = id,
                    # track hyperparameters and run metadata
                    config= config
                    )
        saving_path = str(f"results/vqe_simulation/jax/Nx-{Nx}_Ny-{Ny}_p-{p}_q-{q}/{id}")
        jax_simulation.vqe_simulation(Nx = Nx, Ny = Ny, config_list = config_list, n = n, p=p, q=q, pre_ansatz = None,saveto = saving_path, log = log)
    
    else: #qiskit 
        extention_factor = config['data']['extention_factor']
        if log:
            id = wandb.util.generate_id()
            config['data']['id'] = id

            wandb.init(
                # set the wandb project where this run will be logged
                project="RFCI-vqe",
                name= f'Nx-{Nx}_Ny-{Ny}_EF-{extention_factor}/{id}',
                id = id,
                # track hyperparameters and run metadata
                config= config
                )
        saving_path = str(f"results/vqe_simulation/Nx-{Nx}_Ny-{Ny}_EF-{extention_factor}/{config['data']['id']}")
        qiskit_simulation.vqe_simulation(Nx = Nx, Ny = Ny, config_list = config_list, n = n, extention_factor = extention_factor , pre_ansatz = None,saveto = saving_path, log = log)
    