import os.path as op
import os
import glob
import json
import argparse
import os
import numpy as np
import cv2
import sys
import yaml
from tqdm import tqdm
from dnnlib.util import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tools import builder
from utils.config import cfg_from_yaml_file, merge_new_config
from utils import misc
from datasets.io import IO
from datasets.data_transforms import Compose
from tools.inference import inference_single

network_type = 'AdaPoinTr'
data_type = 'TeethSeg_models'
ckpt_type = 'ckpt-best.pth'
overwrite = True
pointr_dir = '/storage/share/repos/PoinTr'

exp_dir = op.join(pointr_dir, 'experiments', network_type, data_type)

model_args = []

for dirpath, dirnames, filenames in os.walk(exp_dir):
    for filename in filenames:
        if filename == ckpt_type:
            args = EasyDict({
                'cfg_name': op.basename(dirpath),
                'model_config': op.join(dirpath, 'config.yaml'), 
                'data_config': op.join(dirpath, 'data-config.yaml'), 
                'model_checkpoint': op.join(dirpath, filename),
                'pc_root': op.join(dirpath, 'inference', 'corr'),
                'out_pc_root': op.join(dirpath, 'inference', 'result'),
                'device': 'cuda:0',
                'save_vis_img': False,    
                })
            model_args.append(args)

for args in model_args:
    

    config = EasyDict()

    with open(args.model_config, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

    for i in ['train', 'val', 'test']:
        new_config['dataset'][i]['_base_'] = args.data_config

    merge_new_config(config=config, new_config=new_config)

    data_config = cfg_from_yaml_file(args.data_config)

    with open(data_config.CATEGORY_FILE_PATH) as f:
        dataset_categories = json.loads(f.read())   

    dataset_categories = [dc for dc in dataset_categories if dc['taxonomy_name'] in data_config.CATEGORY_DICT.keys()]

    file_list = []
    partial_paths = []
    for dc in dataset_categories:
        samples = dc['test']
        for s in samples:
            file_list.append({
                'taxonomy_id':  dc['taxonomy_id'],
                'model_id':     s,
                'partial_path': [data_config.PARTIAL_POINTS_PATH % ('/'.join(dc['taxonomy_name'].split('_')), data_config.N_POINTS_PARTIAL, s, tooth) for tooth in data_config.CATEGORY_DICT[dc['taxonomy_name']]],
                'gt_path':      data_config.COMPLETE_POINTS_PATH % ('/'.join(dc['taxonomy_name'].split('_')), data_config.N_POINTS_GT, s),
            })
            partial_paths.extend(file_list[-1]['partial_path'])

    if (not op.exists(args.out_pc_root) or len(os.listdir(args.out_pc_root)) != len(partial_paths)) or overwrite:
        print('\n# -------------------------------------------------------------------------------------------------- #')
        print(args.cfg_name, '\n')

        base_model = builder.model_builder(config.model)

        builder.load_model(base_model, args.model_checkpoint)

        base_model.to(args.device.lower())
        base_model.eval()

        for corr_file in tqdm(partial_paths):
            parts = corr_file.split(op.sep)
            patient = parts[-2]
            num_points = parts[-3]
            toothrange = parts[-4]
            jaw = parts[-5]
            tooth = parts[-1].split('.')[0]

            # split the path into parts
            filename = f'{patient}_corr-{jaw}-{toothrange}_recon-{tooth}_npoints-{data_config.N_POINTS}'

            inference_single(base_model, corr_file, args, config, save_as_pcd=True, filename=filename)
    else:
        print(f'Files for {args.cfg_name} already exist.')
        
