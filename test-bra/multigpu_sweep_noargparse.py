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
import wandb
import pprint
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
import argparse


wandb.init(project="test")

device = torch.device("cuda") #if torch.cuda.is_available() else "cpu")

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)


        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})           

def build_dataset(batch_size):
   
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # download MNIST training dataset
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader


def build_network(fc_layer_size, dropout):
    network = nn.Sequential(  # fully-connected, single hidden layer
        nn.Flatten(),
        nn.Linear(784, fc_layer_size), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_layer_size, 10),
        nn.LogSoftmax(dim=1))

    return network.to(device)
        

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer


def train_epoch(network, loader, optimizer):
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forward pass
        loss = F.nll_loss(network(data), target)
        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)


# change to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sweep_configuration = {
    "description": "Sweep test",
    "program": "wandb-sasdweep.py",
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
    wandb.agent(sweep_id, train)


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
