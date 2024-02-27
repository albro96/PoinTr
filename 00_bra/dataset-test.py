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
from datasets.TeethSegDataset import TeethSeg


params = [
        {'lower_1-7': 'all'},
        {'lower_1-3': 'all'},
        {'lower_35-37': ['36']},
        {'lower_1-7': ['36', '46']},
    ]

param = params[0]

data_config = EasyDict({
        "NAME": "TeethSeg",
        "N_POINTS_GT": 4096,
        "N_POINTS_PARTIAL": 8192, # 2048 4096 8192 16384
        "CATEGORY_DICT":{'jaw': 'lower', 'toothrange': '1-7', 'recon': 'all'}, # 'all' or list of tooth numbers as strings ["36"]
        "GT_TYPE": "single",
        "CORR_TYPE": "corr", # "corr" or "corr-concat" for concat select n_points_partial per tooth
        "DATA_DIR": pada.datasets.TeethSeg22.data_dir,
        "DATA_FILTER_PATH": pada.datasets.TeethSeg22.data_filter_path
        "SAMPLING_METHOD": 'None', # 'None' 'RandomSamplePoints', 'FurthestPointSample'
    })

dataset = TeethSeg(data_config, 'train')

pprint(dataset.file_list)