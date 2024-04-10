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
import open3d as o3d

sys.path.append('/storage/share/code/01_scripts/modules/')
from os_tools.import_dir_path import import_dir_path

pada = import_dir_path()

sys.path.append(pada.models.pointr.repo_dir)
from tools import builder
from tools.inference import inference_single
from datasets.TeethSegDataset import TeethSegDataset

model_name = 'cool-sweep-3'

ckpt_types = ['ckpt-best'] #
device = torch.device('cuda:0')

for ckpt_type in ckpt_types:

    if ckpt_type.endswith('.pth'):
        ckpt_type = Path(ckpt_type).stem

    model_dir = pada.models.pointr.model_dir
    overwrite = True

    model_args = []

    dataset_type = 'train'
    num_items = 10

    # find the file that ends with f'{model_name}.pth' in model_dir recursively

    for dirpath, dirnames, filenames in os.walk(model_dir):
        
        for filename in filenames:
            # print(filename)
            if filename.endswith(f'{model_name}.pth'):
                print(dirpath)
                args = EasyDict({
                    'base_dir': op.dirname(dirpath), 
                    'cfg_name': Path(filename).stem,
                    'model_config': op.join(op.dirname(dirpath), 'config', f'config-{model_name}.json'), 
                    'model_checkpoint': op.join(dirpath, filename),
                    'inference_dir': op.join(op.dirname(dirpath), 'inference'),
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

        data_config = config.dataset

        dataset = TeethSegDataset(**data_config, mode=dataset_type)
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1,
            shuffle=False, 
            pin_memory=False,
            num_workers=10)

        state_dict = torch.load(args.model_checkpoint, map_location='cpu')

        suffix = ''

        folder_name = f'{args.cfg_name}_{dataset_type}-epoch-{state_dict["epoch"]}{suffix}'

        args.inference_dir = op.join(args.inference_dir,  folder_name)

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


        if not op.exists(args.inference_dir) or overwrite:
            print('\n# -------------------------------------------------------------------------------------------------- #')
            print(args.cfg_name, '\n')
            print('Inference on test set\n')
            os.makedirs(args.inference_dir, exist_ok=True)

            shutil.copy(__file__, args.inference_dir)

            base_model = builder.model_builder(config.model)

            builder.load_model(base_model, args.model_checkpoint)

            base_model.to(args.device.lower())
            base_model.eval()

            for idx, (corr, gt) in enumerate(data_loader):

                if idx == num_items:
                    break
        
                filename = str(idx).zfill(3)

                t0 = time.time()

                ret = base_model(corr.to(args.device.lower()))
                pred = ret[-1] #.squeeze(0).detach().cpu().numpy()

                corr = corr.detach().cpu()
                gt = gt.detach().cpu()
                pred = pred.detach().cpu()
                
                for data, name in zip([corr, gt, pred], ['corr', 'gt', 'pred']):
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(data.squeeze(0).numpy())
                    o3d.io.write_point_cloud(op.join(args.inference_dir, f'{filename}-{name}.pcd'), pcd)

                    if name != 'corr':
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(torch.cat([data, corr], dim=1).squeeze(0).numpy())
                        o3d.io.write_point_cloud(op.join(args.inference_dir, f'{filename}-full-{name}.pcd'), pcd)


                print(f'{filename} done in {time.time()-t0:.2f} s')
        else:
            print(f'Files for {args.cfg_name} already exist.')
            