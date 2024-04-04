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
    wandb.agent(sweep_id)

if __name__ == '__main__':
    pada = import_dir_path()

    sweep_configuration = {
    "description": "Sweep test",
    "program": "run_training.py",
    "method": "random",
    "metric": {
        "name": "test/CDL2",
        "goal": "minimize"
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10
    },
    "parameters": {
        "optimizer.type": {
            "values": ['AdamW', 'SGD', 'Adam']
        },
        "optimizer.kwargs.lr": {
            "distribution": "uniform",
            "min": 0.00001,
            "max": 0.01
        },
        "optimizer.kwargs.weight_decay": {
            "distribution": "uniform",
            "min": 0.00001,
            "max": 0.01
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
            "distribution": "int_uniform",
            "min": 0,
            "max": 100
        },
        'scheduler.kwargs.lr_decay': {
            "distribution": "uniform",
            "min": 0.02,
            "max": 0.9
        },
        'bnmscheduler.kwargs.bn_decay': {
            "distribution": "uniform",
            "min": 0.01,
            "max": 0.9
        },
        'bnmscheduler.kwargs.bn_momentum': {
            "distribution": "uniform",
            "min": 0.5,
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

