#%%
from argparse import ArgumentParser
import yaml
from exact_diagnolization import *
from qiskit_simulation import *
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
    print(args.log)
    
    config_file = args.config_path
    # config_file = './config.yaml'
    with open(config_file, 'r') as stream:
            config = yaml.load(stream, yaml.FullLoader)
    config_list_keys = list(config)
    config_list = [config[key] for key in config_list_keys if key is not None and 'config' in key]
    log = args.log

    Nx = config['data']['Nx']
    Ny = config['data']['Ny']
    extention_factor = config['data']['extention_factor']
    n = config['data']['n']

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
    vqe_simulation(Nx = Nx, Ny = Ny, config_list = config_list, n = n, extention_factor = extention_factor , pre_anzats = None,saveto = saving_path, log = log)
    