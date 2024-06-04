import os.path as op
import os
import json
import sys
from tqdm import tqdm
from easydict import EasyDict
from pathlib import Path
import torch
import shutil
import time
from pprint import pprint
import numpy as np

sys.path.append("/storage/share/code/01_scripts/modules/")
from os_tools.import_dir_path import import_dir_path, convert_path

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


pada = import_dir_path()

sys.path.append(pada.models.pointr.repo_dir)
from datasets.TeethSegDataset import TeethSegDataset

import open3d as o3d

data_config = EasyDict(
    {
        "num_points_gt": 8192,  # 2048, #2048,
        "num_points_corr": 0,  # 16384, #16384,  # 2048 4096 8192 16384
        "num_points_corr_type": "full",
        "num_points_gt_type": "full",
        "tooth_range": {
            "corr": "full",
            "gt": "full",  # "full",
            "jaw": "full-separate",
            "quadrants": "all",
        },
        "return_only_full_gt": True,
        "gt_type": "full",
        "data_type": "npy",
        "samplingmethod": "fps",
        "downsample_steps": 2,
        "use_fixed_split": False,
        "splits": {"train": 1, "val": 0, "test": 0},
        "enable_cache": False,
        "create_cache_file": True,
        "overwrite_cache_file": False,
        "cache_dir": op.join(
            pada.base_dir, "nobackup", "data", "3DTeethSeg22", "cache"
        ),
    }
)

# from pcd_tools.visualizer import ObjectVisualizer

save_dir = convert_path(r"O:\nobackup\data\3DTeethSeg22\testdata-lower-16384-fullgt")

os.makedirs(save_dir, exist_ok=True)

if __name__ == "__main__":

    device = torch.device("cuda:0")

    train_set = TeethSegDataset(**data_config, mode="train", device=device)

    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    print(len(loader))
    # for idx, data in enumerate(loader):
    #     corr = data[0]

    # gt = data[1]
    # print(data[0, :10, :])
    # break
    # print(data.shape)
    # # save pcds
    # pcd_corr = o3d.geometry.PointCloud()
    # pcd_corr.points = o3d.utility.Vector3dVector(corr[0].cpu().numpy())
    # o3d.io.write_point_cloud(
    #     op.join(
    #         save_dir,
    #         f"{idx:04d}_{val_loader.dataset.patient}-{val_loader.dataset.tooth}_corr.pcd",
    #     ),
    #     pcd_corr,
    # )

    # pcd_gt = o3d.geometry.PointCloud()
    # pcd_gt.points = o3d.utility.Vector3dVector(gt[0].cpu().numpy())
    # o3d.io.write_point_cloud(
    #     op.join(
    #         save_dir,
    #         f"{idx:04d}_{val_loader.dataset.patient}-{val_loader.dataset.tooth}_gt.pcd",
    #     ),
    #     pcd_gt,
    # )

    # for idx, data in enumerate(val_loader):
    #     print(data[0].shape, data[1].shape)

    #     print(val_loader.dataset.patient, val_loader.dataset.tooth)

    # pcd_corr = o3d.geometry.PointCloud()
    # pcd_corr.points = o3d.utility.Vector3dVector(data[0].cpu().numpy())
    # pcd_gt = o3d.geometry.PointCloud()
    # pcd_gt.points = o3d.utility.Vector3dVector(data[1].cpu().numpy())

    # # save pcds
    # o3d.io.write_point_cloud(
    #     op.join(save_dir, f"{idx:04d}_corr.pcd"), pcd_corr
    # )

    # o3d.io.write_point_cloud(op.join(save_dir, f"{idx:04d}_gt.pcd"), pcd_gt)

    # full = data[0].cpu().numpy()

    # train_loader = torch.utils.data.DataLoader(
    #         train_set,
    #         batch_size=1,
    #         shuffle=False,
    #         pin_memory=False,
    #         num_workers=0)

    # for epoch in range(1):
    #     t0 = time.time()
    #     for i, data in enumerate(train_loader):
    #         # gt = data[0].to(device)
    #         # corr = data[1].to(device)
    #         print(data[0].shape, data[1].shape)

    #         pcd = o3d.geometry.PointCloud()
    #         pcd.points = o3d.utility.Vector3dVector(data[0][0])
    #         visu = ObjectVisualizer()
    #         visu.load_obj(obj=pcd, dtype='pcd')
    #         visu.show()
    #         # visu.screenshot(save_path='/storage/share/nobackup/data/3DTeethSeg22/test.png')
    #         break

    #         # visulist = []
    #         # for num in range(2):
    #         #     pcd = o3d.geometry.PointCloud()
    #         #     pcd.points = o3d.utility.Vector3dVector(data[num][0])
    #         #     # visulist.append(pcd)
    #         #     visulist = [pcd]
    #         #     o3d.visualization.draw_geometries(visulist)


# params = [
#         {'lower_1-7': 'all'},
#         {'lower_1-3': 'all'},
#         {'lower_35-37': ['36']},
#         {'lower_1-7': ['36', '46']},
#     ]

# param = params[0]

# data_config = EasyDict({
#         "NAME": "TeethSeg",
#         "N_POINTS_GT": 4096,
#         "N_POINTS_PARTIAL": 8192, # 2048 4096 8192 16384
#         "CATEGORY_DICT":{'jaw': 'lower', 'toothrange': '1-7', 'recon': 'all'}, # 'all' or list of tooth numbers as strings ["36"]
#         "GT_TYPE": "single",
#         "CORR_TYPE": "corr", # "corr" or "corr-concat" for concat select n_points_partial per tooth
#         "DATA_DIR": pada.datasets.TeethSeg22.data_dir,
#         "DATA_FILTER_PATH": pada.datasets.TeethSeg22.data_filter_path,
#         "SAMPLING_METHOD": 'None', # 'None' 'RandomSamplePoints', 'FurthestPointSample'
#     })

# dataset = TeethSeg(data_config, 'train')

# pprint(dataset.file_list)
