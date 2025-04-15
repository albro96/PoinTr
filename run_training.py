import time
import os
import torch
import sys
from pathlib import Path
import torch.multiprocessing as mp
import json
import shutil
from datetime import datetime
from easydict import EasyDict
import os.path as op
import wandb
import argparse
from pprint import pprint
import copy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../"))
sys.path.append("/storage/share/code/01_scripts/modules/")

from os_tools.import_dir_path import import_dir_path, convert_path

from tools import run_net
from tools import test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *

# ---------------------------------------- #
# ----------------- Main ----------------- #
# ---------------------------------------- #


def main(rank=0, world_size=1):
    pada = import_dir_path()

    data_config = EasyDict(
        {
            "num_points_gt": 2048,
            "num_points_corr": 16384, # 16384, 28000
            "num_points_corr_anta": 1024,
            "num_points_corr_type": "full",
            "num_points_gt_type": "full",
            "tooth_range": {
                "corr": "full",
                "gt":  [6], #[6],  5# # full"
                "jaw": "full",
                "quadrants": "all",
            },
            "return_only_full_gt": False,
            "gt_type": "single",
            "data_type": "npy",
            "samplingmethod": "fps",
            "downsample_steps": 2,
            "use_fixed_split": True,
            "enable_cache": True,
            "create_cache_file": True,
            "overwrite_cache_file": False,
            "return_antagonist": True,
            "return_normals": True,
            "dataset": "orthodental",
            "data_dir": None,
            "datahash": '15c02eb0',
            "gingiva": True,
            "crop_anta": True,
            'crop_anta_thresh': 1,
            'normalize_pose': False,
            'normalize_scale': False,
        }
    )

    args = EasyDict(
        {
            "launcher": "pytorch" if world_size > 1 else "none",
            "num_gpus": world_size,
            "local_rank": rank,
            "num_workers": 0,  # only applies to mode='train', set to 0 for val and test
            "seed": 0,
            "deterministic": False,
            "sync_bn": False,
            "experiment_dir": pada.model_base_dir,
            "start_ckpts": None,
            "val_freq": 10,
            "resume": False,
            "mode": None,
            "save_checkpoints": True,
            "save_only_best": False,
            "ckpt_dir": None,
            "cfg_dir": None,
            'log_testdata': True,
            'ckpt_path': convert_path(r"O:\data\models\PoinTr\sweep\PoinTr-InfoCD-CD-downsample1\ckpt\ckpt-best-zany-sweep-2.pth"),
            "gt_partial_saved": False,
            'no_occlusion_val': 100,
            "test": False,
            "log_data": True,  # if true: wandb logger on and save ckpts to local drive
        }
    )

    config = EasyDict(
        {
        'loss': {
                'cd_type': 'CDL2', # 'CDL2', 'InfoCDL2'
                "active_metrics": {
                    "SparseLoss": True,
                    "DenseLoss": True,
                    'KLVLoss': True,
                    "OcclusionLoss": False,
                    "ClusterDistLoss": False,
                    "ClusterNumLoss": False,
                    "ClusterPosLoss": False,
                    "PenetrationLoss": False,
                },
                "coeffs": {
                    "SparseLoss": 1.0,
                    "DenseLoss": 1.0,
                    'KLVLoss': 1.0,
                    "OcclusionLoss": 1.0,
                    "ClusterDistLoss": 1.0,
                    "ClusterPosLoss": 1.0,
                    "ClusterNumLoss": 1.0,
                    "PenetrationLoss": 1.0,
                },
                "adaptive": True,
                "adaptive_thresh": 2.0,
                "adaptive_steepness": 2.0,  # higher values more abrupt change around adaptive_loss_thresh
                'InfoCD': {
                    'tau': 0.1,
                    'point_reduction': 'sum',
                    'square': False,
                    'two_sided_reduction': 'mean',
                },
            },
            "optimizer": {
                "type": "AdamW",
                "kwargs": {
                    "lr": 0.0001,
                    "weight_decay": 0.001,  # 0.0001
                },
            },
            "scheduler": {
                "type": "LambdaLR",
                "kwargs": {
                    "decay_step": 50,  # 40,
                    "lr_decay": 0.7,  # 0.7,
                    "lowest_decay": 0.02,  # min lr = lowest_decay * lr
                },
            },
            "bnmscheduler": {
                "type": "Lambda",
                "kwargs": {
                    "decay_step": 40,
                    "bn_decay": 0.5,
                    "bn_momentum": 0.9,
                    "lowest_decay": 0.01,
                },
            },
            "dataset": data_config,
            "model": {
                "gt_type": data_config.gt_type,
            },
            "val_metrics": [
                "CDL1",
                "CDL2",
                # "F-Score",
                # "OcclusionLoss",
                # "ClusterDistLoss",
                # "ClusterNumLoss",
                # "ClusterPosLoss",
                # "PenetrationLoss",
                "InvIOULoss",
            ],
            "occlusion_metrics": ['OcclusionLoss', 'ClusterDistLoss', 'ClusterNumLoss', 'ClusterPosLoss', 'PenetrationLoss'],
            "total_bs": int(256 * world_size),  # CRAPCN: int(30*world_size),
            "step_per_update": 1,
            "grad_norm_clip": 5,
            "model_name": "DiscreteVAE",  # "DiscreteVAE" "PoinTr", 'CRAPCN'
            'datasettype': 'TeethSegDataset', # TSTDataset
            "max_epoch": 500,
            "consider_metric": "CDL2",  # "CDL2",
        }
    )

    network_config_dict = EasyDict()

    network_config_dict.AdaPoinTr = {
        "model": {
            "NAME": "AdaPoinTr",
            "num_query": 512,
            "num_points": data_config.num_points_gt,
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
                "block_style_list": [
                    "attn-graph",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                ],
                "combine_style": "concat",
            },
            "decoder_config": {
                "embed_dim": 384,
                "depth": 8,
                "num_heads": 6,
                "k": 8,
                "n_group": 2,
                "mlp_ratio": 2.0,
                "self_attn_block_style_list": [
                    "attn-graph",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                ],
                "self_attn_combine_style": "concat",
                "cross_attn_block_style_list": [
                    "attn-graph",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                    "attn",
                ],
                "cross_attn_combine_style": "concat",
            },
        },
    }

    network_config_dict.DiscreteVAE = {
        "model": {
        "NAME": "DiscreteVAE",
        "group_size": 32,
        "num_group": 64,
        "encoder_dims": 256,
        "num_tokens": 8192,
        "tokens_dims": 256,
        "decoder_dims": 256
        },
        'bnmscheduler': None,
        "scheduler": {
            "type": "CosLR",
            "kwargs": {
                "t_max": 500, # former epochs
                "initial_epochs": 10,
                "warmup_lr_init": 1e-5,
                'min_lr': 0.00001,
            }
            },
        "temp": {
        "start": 1,
        "target": 0.0625,
        "ntime": 100000
        },
        "kldweight": {
            "start": 0,
            "target": 0.1,
            "ntime": 100000
        },   
        # 'loss_metrics': ['Loss1', 'Loss2'],
        'datasettype': 'SingleToothDataset',
        'dataset':{
            "num_points": 2048,
            "tooth_range": {
                "teeth": 'full', 
                "jaw": "full",
                "quadrants": "all",
            },
            "data_type": "npy",
            "use_fixed_split": True,
            "enable_cache": True,
            "create_cache_file": True,
            "overwrite_cache_file": False,
            "return_normals": False, 
            "dataset": "orthodental",
            "data_dir": None,
            "datahash": "15c02eb0",
            'normalize_mean': True,
            'normalize_pose': False,
            'normalize_scale': False,
        }}

    network_config_dict.PoinTr = {
        "model": {
            "NAME": "PoinTr",
            "num_pred": data_config.num_points_gt,
            "num_query": data_config.num_points_gt // 64,  # num_points_gt // 64,
            "knn_layer": 1,
            "trans_dim": 384,  # stays 384 in all configs
        },
    }

    network_config_dict.PCN = {
        "model": {
            "NAME": "PCN",
            "num_pred": data_config.num_points_gt,
            "encoder_channel": 1024,
        },
    }

    network_config_dict.CRAPCN = {
        "model": {
            "NAME": "CRAPCN",
            "loss_subsample": True,  # calc both direction CD (True )for the sparse PCDs Pc P1 P2 or single direction (False)
        },
    }

    if args.test and args.resume:
        raise ValueError("--test and --resume cannot be both activate")

    if args.resume and args.start_ckpts is not None:
        raise ValueError("--resume and --start_ckpts cannot be both activate")


    if args.local_rank is not None:
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(args.local_rank)

    # CUDA
    args.use_gpu = torch.cuda.is_available()
    args.use_amp_autocast = False
    args.device = torch.device("cuda" if args.use_gpu else "cpu")

    # args.device = torch.device('cuda', args.local_rank) if args.use_gpu else torch.device('cpu')
    # print('\n\nHJKASFLDAS\n\n\n')
    # print(args.device)

    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(launcher=args.launcher, rank=args.local_rank)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size

    # exit()
    print(f"Distributed training: {args.distributed}")

    # set random seeds
    if args.seed is not None:
        print(
            f"Set random seed to {args.seed}, " f"deterministic: {args.deterministic}"
        )
        misc.set_random_seed(
            args.seed + args.local_rank, deterministic=args.deterministic
        )  # seed + rank, for augmentation

    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    if args.test:
        assert args.ckpt_path is not None, "Please provide ckpt_path for testing"
        model_name = "-".join(op.basename(args.ckpt_path).split('.')[0].split('-')[2:5])
        cfg_path = op.join(Path(args.ckpt_path).parent.parent, "config", f'config-{model_name}.json')
        with open(cfg_path, "r") as json_file:
            config = EasyDict(json.load(json_file))
        pprint(config)
        args.experiment_path = os.path.join(args.experiment_dir, config.model_name)
        test_net(args, config)
        return
    
    if args.local_rank == 0:
        wandb_config = None

        if args.log_data:
            
            wandb.init(
                # set the wandb project where this run will be logged, dont set config here, else sweep will fail
                project="ToothRecon",
                save_code=True,
            )

            # define custom x axis metric
            wandb.define_metric("epoch")

            # set all other train/ metrics to use this step
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("test/*", step_metric="epoch")

            wandb.define_metric("val/pcd/dense/*", step_metric="epoch")
            wandb.define_metric("val/pcd/full-dense/*", step_metric="epoch")
            wandb.define_metric("val/pcd/gt/*", step_metric="epoch")

            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            wandb_config = wandb.config

            # update the model config with wandb config
            for key, value in wandb_config.items():
                if "." in key:
                    keys = key.split(".")
                    config_temp = config
                    for sub_key in keys[:-1]:
                        config_temp = config_temp.setdefault(sub_key, {})
                    config_temp[keys[-1]] = value
                else:
                    config[key] = value

        if wandb_config is not None:
            args.sweep = True if "sweep" in wandb_config else False
        else:
            args.sweep = False
    else:
        args.sweep = False
        os.environ["WANDB_MODE"] = "disabled"

    # config.model.update(network_config_dict[config.model_name].model)
    config.update(network_config_dict[config.model_name])

    if config.model.NAME == "AdaPoinTr":
        config.model.dense_loss_coeff = config.dense_loss_coeff

    args.log_data = args.sweep or args.log_data

    args.experiment_path = os.path.join(args.experiment_dir, config.model_name)

    if args.sweep:
        args.experiment_path = os.path.join(
            args.experiment_path, "sweep", config.sweepname
        )

    if args.log_data and not args.test and args.local_rank == 0:
        if not os.path.exists(args.experiment_path):
            os.makedirs(args.experiment_path, exist_ok=True)
            print("Create experiment path successfully at %s" % args.experiment_path)

        shutil.copy(__file__, args.experiment_path)

        # set the wandb run dir
        # wandb.run.dir = args.experiment_path

        args.cfg_dir = op.join(args.experiment_path, "config")
        args.ckpt_dir = op.join(args.experiment_path, "ckpt")
        os.makedirs(args.cfg_dir, exist_ok=True)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        cfg_name = f"config-{wandb.run.name}.json"

        with open(os.path.join(args.cfg_dir, cfg_name), "w") as json_file:
            json_file.write(json.dumps(config, indent=4))
        
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.bs = config.total_bs // world_size
    else:
        config.bs = config.total_bs

    if config.dataset.get("num_points_corr",0) > 16384:
        print('\nBATCH SIZE\nBatch size set to 1 due to large number of points\n')
        config.bs = 1

    config['loss_metrics'] = [key for key, value in config.loss.active_metrics.items() if value]

    if config.consider_metric not in config.val_metrics: # or (config.loss.adaptive and not any([metric in config.loss_metrics for metric in occlusion_metrics])):
        print('Invalid Config')
        raise ValueError('Invalid Config')

    config.loss.adaptive = config.loss.adaptive and any([metric in config.loss_metrics for metric in config.occlusion_metrics])

    if args.log_data and args.local_rank == 0:
        # update wandb config
        wandb.config.update(config, allow_val_change=True)

    pprint(config)

    torch.autograd.set_detect_anomaly(True)
    run_net(args, config)


# ---------------------------------------- #
# ----------------- RUN ------------------ #
# ---------------------------------------- #
if __name__ == "__main__":

    # User Input
    num_gpus = 4  # number of gpus, dont use 3
    print("Number of GPUs: ", num_gpus)

    if num_gpus > 1:

        if num_gpus == 2:
            os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        elif num_gpus == 3:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
        elif num_gpus == 4:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"  # Set any free port
        os.environ["WORLD_SIZE"] = str(num_gpus)
        # mp.spawn(main, args=(num_gpus, ), nprocs=num_gpus, join=True)
        mp.spawn(main, args=(num_gpus,), nprocs=num_gpus, join=True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        main(rank=0, world_size=1)
