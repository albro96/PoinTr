import os.path as op
import os
import json
import sys
from tqdm import tqdm
from easydict import EasyDict
from pathlib import Path
import torch
import shutil

# sys.path.append('/storage/share/code/01_scripts/modules/')

from os_tools.import_dir_path import import_dir_path

pada = import_dir_path()

sys.path.append(pada.models.pointr.repo_dir)
from tools import builder
from tools.inference import inference_single
from datasets.TeethSegDataset import TeethSeg

model_name = '240117_PoinTr_lower_1-7--all-corr-8192_gt-single-4096_denseloss'

ckpt_types = ['ckpt-epoch-100'] #

for ckpt_type in ckpt_types:

    if ckpt_type.endswith('.pth'):
        ckpt_type = Path(ckpt_type).stem

    model_dir = pada.models.pointr.model_dir
    overwrite = False

    model_args = []

    dataset_type = 'test'

    for dirpath, dirnames, filenames in os.walk(model_dir):
        for filename in filenames:
            if Path(filename).stem == ckpt_type:
                args = EasyDict({
                    'cfg_name': op.basename(dirpath),
                    'model_config': op.join(dirpath, 'config.json'), 
                    'model_checkpoint': op.join(dirpath, filename),
                    'pc_root': op.join(dirpath, 'inference', 'corr'),
                    'out_pc_root': op.join(dirpath, 'inference'),
                    'device': 'cuda:0',
                    'save_vis_img': False,    
                    })
                model_args.append(args)

    for args in model_args:
        if model_name not in args.cfg_name:
            continue
        
        # load the config json
        with open(args.model_config, 'r') as f:
            config = EasyDict(json.load(f))

        data_config = config.dataset.train._base_

        dataset = TeethSeg(data_config, dataset_type)

        file_list = dataset.file_list


        partial_paths = []
        for sample in file_list:
            if type(sample['partial_path']) == list:
                partial_paths.extend(sample['partial_path'])
            else:
                partial_paths.append(sample['partial_path'])


        # with open(data_config.CATEGORY_FILE_PATH) as f:
        #     dataset_categories = json.loads(f.read())   

        # dataset_categories = [dc for dc in dataset_categories if dc['taxonomy_name'] in data_config.CATEGORY_DICT.keys()]

        # file_list = []
        # partial_paths = []
        # for dc in dataset_categories:
        #     samples = dc[dataset_type]
        #     for s in samples:
        #         file_list.append({
        #             'taxonomy_id':  dc['taxonomy_id'],
        #             'model_id':     s,
        #             'partial_path': [data_config.PARTIAL_POINTS_PATH % ('/'.join(dc['taxonomy_name'].split('_')), data_config.N_POINTS_PARTIAL, s, tooth) for tooth in data_config.CATEGORY_DICT[dc['taxonomy_name']]],
        #             'gt_path':      data_config.COMPLETE_POINTS_PATH % ('/'.join(dc['taxonomy_name'].split('_')), data_config.N_POINTS_GT, s),
        #         })
        #         partial_paths.extend(file_list[-1]['partial_path'])


        state_dict = torch.load(args.model_checkpoint, map_location='cpu')

        # if 'best' in Path(args.model_checkpoint).stem:
        #     suffix = '-best'
        # elif 'last' in Path(args.model_checkpoint).stem:
        #     suffix = '-last'
        # else:
        suffix = ''

        folder_name = f'{dataset_type}-epoch-{state_dict["epoch"]}{suffix}'

        args.out_pc_root = op.join(args.out_pc_root,  folder_name)

        # save metrics for ckpt
        metrics_path = op.join(op.dirname(args.model_checkpoint), 'metrics.json')

        if op.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = EasyDict(json.load(f))
        else:
            metrics = EasyDict()

        if metrics.get(f'Epoch-{state_dict["epoch"]}{suffix}', None) is None:
            metrics[f'Epoch-{state_dict["epoch"]}{suffix}'] = state_dict['metrics']
            # save as json indent 4
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)

        # print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)

        if (not op.exists(args.out_pc_root) or len(os.listdir(args.out_pc_root)) != len(partial_paths)+1) or overwrite:
            print('\n# -------------------------------------------------------------------------------------------------- #')
            print(args.cfg_name, '\n')

            os.makedirs(args.out_pc_root, exist_ok=True)

            shutil.copy(__file__, args.out_pc_root)

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

                inference_single(
                    model = base_model, 
                    pc_path = corr_file, 
                    args = args,
                    config = config, 
                    save_as_pcd=True, 
                    filename=filename,
                    data_config = data_config
                    )
        else:
            print(f'Files for {args.cfg_name} already exist.')
            