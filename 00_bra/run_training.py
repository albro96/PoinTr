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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append('/storage/share/code/01_scripts/modules/')

from os_tools.import_dir_path import import_dir_path

from tools import run_net
from tools import test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *


def main(rank, world_size):
    pada = import_dir_path()

    data_config = EasyDict({
        "NAME": "TeethSeg",
        "CATEGORY_FILE_PATH": "/storage/share/data/3d-datasets/3DTeethSeg22/categories/TeethSeg.json",
        "N_POINTS": 8192,
        "N_POINTS_GT": 8192,
        "N_POINTS_PARTIAL": 8192,
        "CATEGORY_DICT": {'lower_35-37': ['36']},
        "PARTIAL_POINTS_PATH": "/storage/share/nobackup/data/3DTeethSeg22/data/corr/%s/%s/%s/%s.pcd",
        # "COMPLETE_POINTS_PATH": "/storage/share/nobackup/data/3DTeethSeg22/data/full/%s/%s/%s.pcd",
        "COMPLETE_POINTS_PATH": "/storage/share/nobackup/data/3DTeethSeg22/data/gt/%s/%s/%s.pcd"
    })

    args = EasyDict({
        # 'config': './cfgs/TeethSeg_models/AdaPoinTr.yaml',  # replace with your actual config file
        'launcher': 'pytorch',
        'local_rank': rank,
        'num_workers': 4,
        'seed': 0,
        'deterministic': False,
        'sync_bn': False,
        'exp_name': datetime.now().strftime('%y%m%d') + '_' + 'corr-35-37-8192_gt-single-8192_rend-36',
        'experiment_dir': pada.models.pointr.model_dir,
        'start_ckpts': None,
        'ckpts': None,
        'val_freq': 1,
        'resume': False,
        'test': False,
        'mode': None,
    })


    network_config = {
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
            "num_points": data_config.N_POINTS,
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

        "total_bs": 16, #16,
        "step_per_update": 1,
        "max_epoch": 1500,
        "consider_metric": "CDL1"
    }




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
    world_size = 1  # number of gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'  # Set any free port
    os.environ['WORLD_SIZE'] = str(world_size)
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)