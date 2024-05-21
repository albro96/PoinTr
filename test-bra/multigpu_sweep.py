import wandb
import pprint
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
import argparse
import os
import multiprocessing
import sys

# change to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sweep_configuration = {
    "description": "Sweep test",
    "program": "wandb-sweep.py",
    "method": "random",
    "metric": {
        "name": "loss",
        "goal": "minimize"
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10
    },
    "parameters": {
        "optimizer": {
            "values": ['adam', 'sgd']
        },
        "fc_layer_size": {
            "values": [128, 256, 512]
        },
        "dropout": {
            "values": [0.3, 0.4, 0.5]
        },
        "epochs": {
            "value": 100
        },
        "learning_rate": {
            "distribution": "uniform",
            "min": 0,
            "max": 0.1
        },
        "batch_size": {
            "distribution": "q_log_uniform_values",
            "q": 8,
            "min": 16,
            "max": 128
        }
    }
}

# Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="test")

def run_agent(gpu_num):
    """
    Run the agent on the specified GPU.

    Args:
        gpu_num (int): The GPU number to use.

    Returns:
        None
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    wandb.agent(sweep_id)


if __name__ == '__main__':

    gpus = 4 # number of gpus or list of gpus
    
    if isinstance(gpus, int):
        gpus = list(range(gpus))

    processes = []
    for gpu_num in gpus:
        p = multiprocessing.Process(target=run_agent, args=(gpu_num,))
        p.start()
        processes.append(p)
      
    for p in processes:
        p.join()
