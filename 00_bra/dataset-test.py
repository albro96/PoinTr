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

sys.path.append('/storage/share/code/01_scripts/modules/')
from os_tools.import_dir_path import import_dir_path

pada = import_dir_path()

sys.path.append(pada.models.pointr.repo_dir)
from tools import builder
from tools.inference import inference_single
from datasets.TeethSegDataset import TeethSegDataset

if __name__ == '__main__':

    device = torch.device('cuda:0')

    tooth_ranges = [
        {'corr': '1-7', 'gt': [41,1], 'jaw': 'lower', 'quadrants': [3,4]},
    ]
    num_pts = [
        {'gt': 128, 'corr': 128},
        # {'gt': 4096, 'corr': 16384},
    ]

    for tooth_range in tooth_ranges:
        for num_pt in num_pts:
            # toothrange = copy.deepcopy(tooth_range)
            # num_pt = copy.deepcopy(num_pt)
            # print(f'Loading dataset with tooth_range: {tooth_range} and num_points: {num_pt}')
            train_set = TeethSegDataset(
                mode='train',
                tooth_range=tooth_range,
                num_points_corr=num_pt['corr'],
                num_points_gt=num_pt['gt'],
                gt_type='full',
                device=device,
                splits={'train': 0.8, 'val': 0.1},
                enable_cache=False,
                create_cache_file=True
            )

    import open3d as o3d
    from pcd_tools.visualizer import ObjectVisualizer

    train_loader = torch.utils.data.DataLoader(
            train_set, 
            batch_size=1,
            shuffle=False, 
            pin_memory=False,
            num_workers=0)

    for epoch in range(1):
        t0 = time.time()
        for i, data in enumerate(train_loader):
            # gt = data[0].to(device)
            # corr = data[1].to(device)
            print(data[0].shape, data[1].shape)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data[0][0])
            visu = ObjectVisualizer()
            visu.load_obj(obj=pcd, dtype='pcd')
            visu.show()
            # visu.screenshot(save_path='/storage/share/nobackup/data/3DTeethSeg22/test.png')
            break

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