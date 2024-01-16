import time
import os
import torch
import sys
from pathlib import Path
from tensorboardX import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import json
import shutil
from datetime import datetime
from easydict import EasyDict
import os.path as op

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append('/storage/share/code/01_scripts/modules/')

from os_tools.import_dir_path import import_dir_path

from tools import run_net
from tools import test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *


def main(rank, world_size, param):
    network_type = 'PoinTr'

    data_config = EasyDict({
        "NAME": "TeethSeg",
        "CATEGORY_FILE_PATH": "/storage/share/data/3d-datasets/3DTeethSeg22/categories/TeethSeg.json",
        "N_POINTS_GT": 2048,
        "N_POINTS_PARTIAL": 4096, # 2048 4096 8192 16384
        "CATEGORY_DICT": {'lower_1-7': 'all'}, # 'all' or list of tooth numbers as strings ["36"]
        "GT_TYPE": "single",
        "CORR_TYPE": "corr", # "corr" or "corr-concat" for concat select n_points_partial per tooth
        "DATA_DIR": "/storage/share/nobackup/data/3DTeethSeg22/data/",
        "DATAFORMAT": "pcd",
        "SAMPLING_METHOD": 'None', # 'None' 'RandomSamplePoints', 'FurthestPointSample'
    })

    suffix = f'denseloss-{int(param*1000)}_CDL2_sample2048'

    if not suffix.startswith('_') and suffix != '':
        suffix = '_' + suffix

    # concat all elements in data_config.CATEGORY_DICT.keys() to a string with - as separator
    toothrange = ''

    for key, val in data_config.CATEGORY_DICT.items():
        toothrange += f'-{key}'

        if isinstance(val, list):
            toothrange += '--' + "-".join(val)
        else:
            toothrange += f'--{val}'

    if toothrange.startswith('-'):
        toothrange = toothrange[1:]

    experiment_name = datetime.now().strftime('%y%m%d') + f'_{network_type}_' + f'{toothrange}-corr-{data_config.N_POINTS_PARTIAL}_gt-{data_config.GT_TYPE}-{data_config.N_POINTS_GT}{suffix}'

    pada = import_dir_path()

    json_path = op.join(pada.datasets.TeethSeg22.base_dir, 'categories', 'toothlist_dict.json')

    with open(json_path, 'r') as f:
        toothlist_dict = json.load(f)       

    for k,v in data_config.CATEGORY_DICT.items():
        if v == 'all':
            data_config.CATEGORY_DICT[k] = toothlist_dict[k.split('_')[0]][k.split('_')[1]]

    data_config.N_POINTS = data_config.N_POINTS_GT

    args = EasyDict({
        # 'config': './cfgs/TeethSeg_models/AdaPoinTr.yaml',  # replace with your actual config file
        'launcher': 'pytorch' if world_size > 1 else 'none',
        'local_rank': rank,
        'num_workers': 10, #4,
        'seed': 0,
        'deterministic': False,
        'sync_bn': False,
        'exp_name': experiment_name,
        'experiment_dir': pada.models.pointr.model_dir,
        'start_ckpts': None,
        'ckpts': None,
        'val_freq': 1,
        'resume': False,
        'test': False,
        'mode': None,
    })


    network_config_dict = EasyDict()

    network_config_dict.AdaPoinTr = {
        "optimizer": {
            "type": "AdamW",
            "kwargs": {
                "lr": 0.0001,
                "weight_decay": 0.0005
            }
        },
        "scheduler": {
            "type": "LambdaLR",
            "kwargs": {
                "decay_step": 21,
                "lr_decay": 0.9,
                "lowest_decay": 0.02
            }
        },
        "bnmscheduler": {
            "type": "Lambda",
            "kwargs": {
                "decay_step": 21,
                "bn_decay": 0.5,
                "bn_momentum": 0.9,
                "lowest_decay": 0.01
            }
        },
        "dataset": {
            "train": {
                "_base_": data_config,
                "others": {
                    "subset": "train"
                }
            },
            "val": {
                "_base_": data_config,
                "others": {
                    "subset": "test"
                }
            },
            "test": {
                "_base_": data_config,
                "others": {
                    "subset": "test"
                }
            }
        },
        "model": {
            "NAME": "AdaPoinTr",
            "num_query": 512,
            "num_points": data_config.N_POINTS_GT,
            "center_num": [512, 256],
            "global_feature_dim": 1024,
            "encoder_type": "graph",
            "decoder_type": "fc",
            "encoder_config": {
                "embed_dim": 384,
                "depth": 6,
                "num_heads": 6,
                "k": 8,
                "n_group": 2,
                "mlp_ratio": 2.0,
                "block_style_list": ["attn-graph", "attn", "attn", "attn", "attn", "attn"],
                "combine_style": "concat"
            },
            "decoder_config": {
                "embed_dim": 384,
                "depth": 8,
                "num_heads": 6,
                "k": 8,
                "n_group": 2,
                "mlp_ratio": 2.0,
                "self_attn_block_style_list": ["attn-graph", "attn", "attn", "attn", "attn", "attn", "attn", "attn"],
                "self_attn_combine_style": "concat",
                "cross_attn_block_style_list": ["attn-graph", "attn", "attn", "attn", "attn", "attn", "attn", "attn"],
                "cross_attn_combine_style": "concat"
            }
        },

        "total_bs": int(16*world_size), #16,
        "step_per_update": 1,
        "max_epoch": 1000,
        "consider_metric": "CDL1",
        "dense_loss_coeff": 0.1,
    }

    network_config_dict.PoinTr = {
        "optimizer": {
            "type": "AdamW",
            "kwargs": {
                "lr": 0.0001,
                "weight_decay": 0.0001
            }
        },
        "scheduler": {
            "type": "LambdaLR",
            "kwargs": {
                "decay_step": 40,
                "lr_decay": 0.7,
                "lowest_decay": 0.02  # min lr = lowest_decay * lr
            }
        },
        "bnmscheduler": {
            "type": "Lambda",
            "kwargs": {
                "decay_step": 40,
                "bn_decay": 0.5,
                "bn_momentum": 0.9,
                "lowest_decay": 0.01
            }
        },
        "dataset": {
            "train": {"_base_": data_config, "others": {"subset": "train"}},
            "val": {"_base_": data_config, "others": {"subset": "test"}},
            "test": {"_base_": data_config, "others": {"subset": "test"}}
        },
        "model": {
            "NAME": "PoinTr",
            "num_pred": data_config.N_POINTS_GT,
            "gt_type": data_config.GT_TYPE,
            "num_query": 224,   # number of coarse points, dense points = 224*9 = 2016 (always true?)
            "knn_layer": 1,
            "trans_dim": 384
        },
        "total_bs": int(14*world_size), #int(28*num_gpus),
        "step_per_update": 1,
        "max_epoch": 800,
        "consider_metric": "CDL2",
        "dense_loss_coeff": param,
    }

    network_config = network_config_dict[network_type]


    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' +args.mode

    args.experiment_path = os.path.join(args.experiment_dir, args.exp_name)
    args.tfboard_path = os.path.join(args.experiment_dir, '00_TFBoard', args.exp_name)

    args.log_name = args.exp_name

    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path, exist_ok=True)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path, exist_ok=True)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

    shutil.copy(__file__, args.experiment_path)

    # CUDA
    args.use_gpu = torch.cuda.is_available()
    args.use_amp_autocast = False

    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(launcher=args.launcher, rank=args.local_rank)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size


    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None

    # config
    # config = get_config(args, logger = logger)       
    config = EasyDict()
    new_config = network_config
    merge_new_config(config=config, new_config=new_config)     

    with open(os.path.join(args.experiment_path, 'config.json'), "w") as json_file:
        json_file.write(json.dumps(config, indent=4))

    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank() 

    # run
    if args.test:
        test_net(args, config)
    else:
        run_net(args, config, train_writer, val_writer)



if __name__ == '__main__':
    # User Input
    num_gpus = 1 # number of gpus

    for param in [1.0]:

        print(param)

        print('Number of GPUs: ', num_gpus)

        if num_gpus > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12345'  # Set any free port
            os.environ['WORLD_SIZE'] = str(num_gpus)
            mp.spawn(main, args=(num_gpus, param), nprocs=num_gpus, join=True)
        else:
            
            main(rank=0, world_size=1, param=param)

    # mp.spawn(main, args=(num_gpus,), nprocs=num_gpus, join=True)