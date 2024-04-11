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

def run_agent(gpu_num, sweep_id, project=None, entity=None,  count=50):
    """
    Run the agent on the specified GPU.

    Args:
        gpu_num (int): The GPU number to use.

    Returns:
        None
    """
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

    print(f'\nRunning agent on GPU: {gpu_num}\n')

    if entity is not None and project is not None:
        wandb.agent(sweep_id=sweep_id, entity=entity, project=project, count=count)
    else:
        wandb.agent(sweep_id, count=count)

if __name__ == '__main__':

    resume_data = {'entity': 'albro96', 'project': 'ToothRecon', 'sweep_id': 'i6puicfz'}
    resume = True


    pada = import_dir_path()

    sweep_configuration = {
    "description": "Only Optimizer/Scheduler",
    "program": "run_training.py",
    "method": "grid", #"random",
    "metric": {
        "name": "val/CDL2",
        "goal": "minimize"
    },
    # 'early_terminate': {
    #     'type': 'hyperband',
    #     'min_iter': 20
    # },
    "parameters": {
        "sweep": {
            "values": [True]
        },
        "dataset.tooth_range.corr": {
            "values": ['35-37', [35, 36, 37, 46], '5-7', '31-37', '1-7']
        },
        "model_name": {
            'values': ['AdaPoinTr', 'PoinTr', 'PCN']
        },

        # "optimizer.type": {
        #     "values": ['AdamW', 'SGD', 'Adam']
        # },
        # "optimizer.kwargs.lr": {
        #     "values": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
        # },
        # "optimizer.kwargs.weight_decay": {
        #     "values": [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
        # },
        # "model.num_query": {
        #     "distribution": "int_uniform",
        #     "min": 128,
        #     "max": 256
        # },
        # "model.knn_layer": {
        #     "distribution": "int_uniform",
        #     "min": 1,
        #     "max": 5
        # },
        # "dense_loss_coeff": {
        #     "distribution": "uniform",
        #     "min": 0,
        #     "max": 10
        # },
        # 'scheduler.kwargs.lr_decay': {
        #     "distribution": "uniform",
        #     "min": 0.1,
        #     "max": 1.0
        # },
        # 'scheduler.kwargs.decay_step': {
        #     "distribution": "int_uniform",
        #     "min": 1,
        #     "max": 50
        # },
        # 'bnmscheduler.kwargs.bn_decay': {
        #     "distribution": "uniform",
        #     "min": 0.1,
        #     "max": 1.0
        # },
        # 'bnmscheduler.kwargs.bn_momentum': {
        #     "distribution": "uniform",
        #     "min": 0.1,
        #     "max": 0.9
        # }
    }
    }

    os.environ['WANDB_DIR'] = pada.model_base_dir

    if not resume:
        # Initialize sweep by passing in config.
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="ToothRecon")
    else:
        print('Resuming sweep with sweep-parameters: ', resume_data)

    gpus = [1,2,3] # number of gpus or list of gpus
    
    if isinstance(gpus, int):
        gpus = list(range(gpus))

    processes = []
    for gpu_num in gpus:
        if not resume:
            p = multiprocessing.Process(target=run_agent, args=(gpu_num,sweep_id))
        else:
            p = multiprocessing.Process(target=run_agent, args=(gpu_num, resume_data['sweep_id'], resume_data['project'], resume_data['entity']))
        p.start()
        processes.append(p)
      
    for p in processes:
        p.join()

