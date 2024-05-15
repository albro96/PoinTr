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

from os_tools.import_dir_path import import_dir_path

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
            "num_points_corr": 16384,  # 2048 4096 8192 16384
            "num_points_corr_type": "full",
            "num_points_gt_type": "single",
            "tooth_range": {
                "corr": "full",
                "gt": "full",
                "jaw": "lower",
                "quadrants": "all",
            },
            "gt_type": "single",
            "data_type": "npy",
            "samplingmethod": "fps",
            "downsample_steps": 2,
            "use_fixed_split": True,
            # "splits": {'train': 0.8,'val': 0.1, 'test': 0.1},
            "enable_cache": True,
            "create_cache_file": True,
            "overwrite_cache_file": False,
            "cache_dir": op.join(
                pada.base_dir, "nobackup", "data", "3DTeethSeg22", "cache"
            ),
        }
    )

    args = EasyDict(
        {
            "launcher": "pytorch" if world_size > 1 else "none",
            "local_rank": rank,
            "num_workers": 16,  # 4,
            "seed": 0,
            "deterministic": False,
            "sync_bn": False,
            "experiment_dir": pada.model_base_dir,
            "start_ckpts": None,
            "ckpts": None,
            "val_freq": 10,
            "test_freq": None,
            "resume": False,
            "test": False,
            "mode": None,
            "save_checkpoints": True,
            "save_only_best": True,
            "ckpt_dir": None,
            "cfg_dir": None,
            "gt_partial_saved": False,
        }
    )

    config = EasyDict(
        {
            "optimizer": {
                "type": "AdamW",
                "kwargs": {
                    "lr": 0.0001,  # royal-sweep-11
                    "weight_decay": 1,  # royal-sweep-11 # 0.0001
                },
            },
            "scheduler": {
                "type": "LambdaLR",
                "kwargs": {
                    "decay_step": 47,  # royal-sweep-11 # 40,
                    "lr_decay": 0.76,  # royal-sweep-11 # 0.7,
                    "lowest_decay": 0.02,  # min lr = lowest_decay * lr
                },
            },
            "bnmscheduler": {
                "type": "Lambda",
                "kwargs": {
                    "decay_step": 40,
                    "bn_decay": 0.96,  # royal-sweep-11 # 0.5,
                    "bn_momentum": 0.55,  # royal-sweep-11 # 0.9,
                    "lowest_decay": 0.01,
                },
            },
            "dataset": data_config,
            "model": {
                "gt_type": data_config.gt_type,
                "cd_norm": 2,
            },
            "max_epoch": 400,
            "consider_metric": "CDL2",
            "total_bs": int(8 * world_size),
            "dense_loss_coeff": 1.0,
            "step_per_update": 1,
            "model_name": "PoinTr",
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

    network_config_dict.PoinTr = {
        "model": {
            "NAME": "PoinTr",
            "num_pred": data_config.num_points_gt,
            "num_query": 222,  # royal-sweep-11 #224,   # number of coarse points, dense points = 224*9 = 2016 (always true?)
            "knn_layer": 2,  # royal-sweep-11 #1,
            "trans_dim": 384,
        },
    }

    network_config_dict.PCN = {
        "model": {
            "NAME": "PCN",
            "num_pred": data_config.num_points_gt,
            "encoder_channel": 1024,
        },
    }

    if args.test and args.resume:
        raise ValueError("--test and --resume cannot be both activate")

    if args.resume and args.start_ckpts is not None:
        raise ValueError("--resume and --start_ckpts cannot be both activate")

    if args.test and args.ckpts is None:
        raise ValueError("ckpts shouldnt be None while test mode")

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

    wandb.init(
        # set the wandb project where this run will be logged
        project="ToothRecon",
        config=config,
        save_code=True,
    )

    # define custom x axis metric
    wandb.define_metric("train/epoch")
    wandb.define_metric("val/epoch")
    wandb.define_metric("test/epoch")

    # set all other train/ metrics to use this step
    wandb.define_metric("train/*", step_metric="train/epoch")
    wandb.define_metric("val/*", step_metric="val/epoch")
    wandb.define_metric("test/*", step_metric="test/epoch")

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

    config.model.update(network_config_dict[config.model_name].model)
    if config.model.NAME == "AdaPoinTr":
        config.model.dense_loss_coeff = config.dense_loss_coeff

    args.sweep = True if "sweep" in wandb_config else False

    args.experiment_path = os.path.join(args.experiment_dir, config.model_name)

    if args.sweep:
        args.experiment_path = os.path.join(
            args.experiment_path, "sweep", wandb.run.sweep_id
        )

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

    # update wandb config
    wandb.config.update(config, allow_val_change=True)

    # pprint(wandb.config)
    pprint(config)

    # run
    if args.test:
        test_net(args, config)
    else:
        run_net(args, config)


# argparse = argparse.ArgumentParser()
# argparse.add_argument('--sweep', type=bool, default=False, help='Sweep mode')
# argparse_args = argparse.parse_args()

# ---------------------------------------- #
# ----------------- RUN ------------------ #
# ---------------------------------------- #
if __name__ == "__main__":

    # User Input
    num_gpus = 1  # number of gpus
    print("Number of GPUs: ", num_gpus)

    if num_gpus > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"  # Set any free port
        os.environ["WORLD_SIZE"] = str(num_gpus)
        # mp.spawn(main, args=(num_gpus, ), nprocs=num_gpus, join=True)
        mp.spawn(main, args=(num_gpus,), nprocs=num_gpus, join=True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        main(rank=0, world_size=1)
