import wandb
import os
import multiprocessing
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append('/storage/share/code/01_scripts/modules/')

from os_tools.import_dir_path import import_dir_path

# change to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_agent(gpu_num, sweep_id):
    """
    Run the agent on the specified GPU.

    Args:
        gpu_num (int): The GPU number to use.

    Returns:
        None
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    wandb.agent(sweep_id, count=50)

if __name__ == '__main__':
    pada = import_dir_path()

    sweep_configuration = {
    "description": "Sweep test",
    "program": "run_training.py",
    "method": "random",
    "metric": {
        "name": "val/CDL2",
        "goal": "minimize"
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10
    },
    "parameters": {
        "sweep": {
            "values": [True]
        },
        "optimizer.type": {
            "values": ['AdamW', 'SGD', 'Adam']
        },
        "optimizer.kwargs.lr": {
            "values": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
        },
        "optimizer.kwargs.weight_decay": {
            "values": [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
        },
        "model.num_query": {
            "distribution": "int_uniform",
            "min": 100,
            "max": 400
        },
        "model.knn_layer": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 10
        },
        "dense_loss_coeff": {
            "distribution": "uniform",
            "min": 0,
            "max": 10
        },
        'scheduler.kwargs.lr_decay': {
            "distribution": "uniform",
            "min": 0.1,
            "max": 1.0
        },
        'scheduler.kwargs.decay_step': {
            "distribution": "int_uniform",
            "min": 1,
            "max": 50
        },
        'bnmscheduler.kwargs.bn_decay': {
            "distribution": "uniform",
            "min": 0.1,
            "max": 1.0
        },
        'bnmscheduler.kwargs.bn_momentum': {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.9
        }
    }
    }

    os.environ['WANDB_DIR'] = pada.models.pointr.model_dir

    # Initialize sweep by passing in config.
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="PoinTr")

    gpus = 4 # number of gpus or list of gpus
    
    if isinstance(gpus, int):
        gpus = list(range(gpus))

    processes = []
    for gpu_num in gpus:
        p = multiprocessing.Process(target=run_agent, args=(gpu_num,sweep_id))
        p.start()
        processes.append(p)
      
    for p in processes:
        p.join()

