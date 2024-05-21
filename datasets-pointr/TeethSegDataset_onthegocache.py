# import torch.utils.data as data
# import numpy as np
import os, sys
# import random
# import os
# import json
# from easydict import EasyDict
# import shutil
# import os.path as op
# import pandas as pd
from tqdm import tqdm
# import trimesh
import time
from pprint import pprint
sys.path.append('/storage/share/code/01_scripts/modules/')
from os_tools.import_dir_path import import_dir_path
# import pcd_tools.data_processing as dp
from general_tools.format import format_duration
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# from .build import DATASETS
# from utils.logger import *
# from .io import IO
# import data_transforms
from hashlib import sha256

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import os.path as op
import json
from easydict import EasyDict
import pytorch3d.ops.sample_farthest_points as fps 
import ctypes
import multiprocessing as mp

# from pytorch3d.structures import Pointclouds, join_pointclouds_as_batch
# from pytorch3d.ops import convert_pointclouds_to_tensor

# mp.set_start_method('spawn')

class TeethSegDataset(Dataset):
    def __init__(
            self, 
            mode='train', 
            jaw='lower', 
            quadrants='all', 
            tooth_range='1-7', 
            num_points_gt=1024, 
            num_points_corr=1024, 
            num_points_orig=8192,
            gt_type='single', 
            data_type='npy',
            splits=None, 
            enable_cache=True, 
            create_cache_file=False,
            overwrite_cache_file=False,
            cache_dir=None,
            device=None
            ):
        # PATHS

        pada = import_dir_path(filepath=__file__)
        self.data_dir = pada.datasets.TeethSeg22.data_single_dir
        self.cache_dir = cache_dir if cache_dir is not None else op.join(pada.base_dir, 'nobackup', 'data', '3DTeethSeg22', 'cache')
        self.create_cache_file = create_cache_file
        self.overwrite_cache_file = overwrite_cache_file
        self.data_type = data_type
        self.data_filter_path = pada.datasets.TeethSeg22.data_filter_path
        self.toothlabels_path = pada.datasets.TeethSeg22.toothlabels_path
        self.device = device if device is not None else torch.device('cpu')
        
        if self.device.type == 'cpu':
            print('Using CPU. This will slow down the data loading process. Consider using CUDA.')

        # USER INPUT
        self.jaw = jaw
        self.quadrants = quadrants
        self.tooth_range = tooth_range
        self.num_points_gt = num_points_gt
        self.num_points_corr = num_points_corr
        self.num_points_orig = num_points_orig
        self.gt_type = gt_type
        self.mode = mode
        self.splits = {'train': 0.8,'val': 0.1, 'test': 0.1} if splits is None else splits

        if 'test' not in self.splits.keys():
            self.splits['test'] = 1 - sum(self.splits.values())

        self.enable_cache = enable_cache      

        if not self.enable_cache:
            print('Caching is disabled. This will slow down the data loading process.\nSet num_workers to 0 in Dataloader if CUDA is required!')

        self.quad_dict = dict(lower=[3,4], upper=[1,2])

        if self.quadrants == 'all':
            self.quadrants = self.quad_dict[self.jaw]

        if self.tooth_range != 'full':
            if isinstance(self.tooth_range, str):
                self.tooth_range_list = [int(self.tooth_range[0]), int(self.tooth_range[-1])]

        self.toothlist = []
        for quadrant in self.quadrants:
            assert quadrant in self.quad_dict[self.jaw]
            self.toothlist.extend([int(f'{quadrant}{tooth}') for tooth in range(self.tooth_range_list[0], self.tooth_range_list[-1]+1)])

        # create data_filter list if it does not exist
        if not op.exists(self.data_filter_path):
            print('Creating new data filter list..')
            data_filter = pd.DataFrame()
            data_filter['patient-all'] = os.listdir(self.data_dir)
            data_filter['patient-filter'] = data_filter.loc[:, 'patient-all']
            data_filter.to_csv(self.data_filter_path, sep=';', index=False)

        # load data_filter list
        self.data_filter = pd.read_csv(filepath_or_buffer=self.data_filter_path, sep=';', usecols=['patient-filter']).dropna().reset_index(drop=True)

        # load the toothdict
        with open(self.toothlabels_path, 'r') as f:
            self.toothdict = EasyDict(json.load(f))

        filterlist = [patient for patient in self.data_filter['patient-filter'] if all(int(elem) in self.toothdict[patient] for elem in self.toothlist)]

        assert np.sum([i for i in self.splits.values()]) == 1, "The sum of split ratios must be equal to 1."

        # Calculate the indices where to split the array
        train_index = int(len(filterlist)*self.splits['train'])
        val_index = train_index + int(len(filterlist)*self.splits['val'])

        # Split the array
        train, val, test = np.split(filterlist, [train_index, val_index])

        # Assign to patientlist based on mode
        if self.mode == 'train':
            self.patientlist = train
        elif self.mode == 'val':
            self.patientlist = val
        elif self.mode == 'test':
            self.patientlist = test
        elif self.mode == 'pick':
            self.patientlist = filterlist

        self.loaded_patients = []

        self.patient_tooth_list = []
        for patient in self.patientlist:
            for tooth in self.toothlist:
                self.patient_tooth_list.append([patient, tooth])
        
        self.num_samples = len(self.patient_tooth_list)

        self.use_cached_data = False

        if self.enable_cache:
            # create hash for cache
            relevant_keys = ['jaw', 'quadrants', 'toothlist', 'num_points_gt', 'num_points_corr', 'num_points_orig', 'gt_type', 'splits', 'mode']

            cache_dict = {key: value for key, value in self.__dict__.items() if key in relevant_keys}
            self.cache_hash = sha256(json.dumps(cache_dict, sort_keys=True).encode()).hexdigest()[:8]
            print(f'Cache hash: {self.cache_hash}')
            self.cache_path = op.join(self.cache_dir, f'{self.cache_hash}.npz')

            shared_array_base_gt = mp.Array(ctypes.c_float, self.num_samples*self.num_points_gt*3)
            shared_array_gt = np.ctypeslib.as_array(shared_array_base_gt.get_obj())
            shared_array_gt = shared_array_gt.reshape(self.num_samples, self.num_points_gt, 3)
            self.shared_array_gt = torch.from_numpy(shared_array_gt)

            shared_array_base_corr = mp.Array(ctypes.c_float, self.num_samples*self.num_points_corr*3)
            shared_array_corr = np.ctypeslib.as_array(shared_array_base_corr.get_obj())
            shared_array_corr = shared_array_corr.reshape(self.num_samples, self.num_points_corr, 3)
            self.shared_array_corr = torch.from_numpy(shared_array_corr)

            if op.exists(self.cache_path) and not self.overwrite_cache_file:
                print(f'Loading cache {self.cache_path}...')
                cache = np.load(self.cache_path)
                self.shared_array_gt = cache['shared_array_gt']
                self.shared_array_corr = cache['shared_array_corr']
                self.use_cached_data = True


    def set_use_cached_data(self, use_cached_data):
            self.use_cached_data = use_cached_data

    def __len__(self):
        if self.use_cached_data or not self.enable_cache:
            return len(self.patient_tooth_list)
        else:
            return len(self.patientlist)

    def resample(self, vertices, num_points):
        vertices = torch.from_numpy(vertices).float().unsqueeze(0).to(self.device)
        vertices = fps(vertices, K=num_points)[0].cpu()
        return vertices
    
    def load_patient_data(self, patient, corr_tooth=None):

        if corr_tooth is not None:
            corr_tooth_idx = self.toothlist.index(corr_tooth)

        #     all_teeth_tensor = torch.empty(0, self.num_points_orig, 3)

        #     for tooth in self.toothlist:
        #         pcd_path = op.join(self.data_dir, f'{self.data_type}_{self.num_points_orig}',patient, f"{tooth}.npy")
        #         tooth_tensor = torch.from_numpy(np.load(pcd_path)).unsqueeze(0)
        #         all_teeth_tensor = torch.cat((all_teeth_tensor, tooth_tensor), dim=0)
        #         if tooth == corr_tooth:
        #             tooth_tensor = tooth_tensor.to(self.device)
        #             gt = fps(tooth_tensor, K=self.num_points_gt)[0][0]

        #     all_teeth_tensor = all_teeth_tensor.to(self.device)
            
        #     corr_tensor = all_teeth_tensor[torch.arange(len(self.toothlist))!=corr_tooth_idx]

        #     # 2 step process: first downsample each downsamples GT tooth to the same number of points, then downsample the whole set
        #     # this is about 4 times faster than downsampling the whole set at once
        #     num_points_corr = torch.ones(len(self.toothlist)-1)*self.num_points_corr/(len(self.toothlist)-1)
        #     num_points_corr = torch.ceil(num_points_corr).int().to(self.device)
        #     corr = fps(corr_tensor, K=num_points_corr)[0].view(1,-1,3)
        #     corr = fps(corr, K=self.num_points_corr)[0][0]

        # else:
        # # CACHING

        # create paths
        all_teeth_tensor = torch.empty(0, self.num_points_orig, 3)
        # start = time.time()
        for tooth in self.toothlist:
            pcd_path = op.join(self.data_dir, f'{self.data_type}_{self.num_points_orig}',patient, f"{tooth}.npy")
            tooth_tensor = torch.from_numpy(np.load(pcd_path)).unsqueeze(0)
            all_teeth_tensor = torch.cat((all_teeth_tensor, tooth_tensor), dim=0)

        # print(f'Loading patient data took {time.time()-start}')

        all_teeth_tensor = all_teeth_tensor.to(self.device)

        # start = time.time()
        gt = fps(all_teeth_tensor, K=self.num_points_gt)[0]

        # print(f'FPS GT took {time.time()-start}')

        # start = time.time()

        corr = torch.empty(0, self.num_points_corr, 3).to(self.device)
        
        for idx, tooth in enumerate(self.toothlist):

            if corr_tooth is not None:
                if idx != corr_tooth_idx:
                    continue

            corr_tensor = gt[torch.arange(len(self.toothlist))!=idx]

            # # downsample the whole set at once
            # corr_tensor = corr_tensor.view(1,-1,3)
            # corr_tensor_sampled = fps(corr_tensor, K=self.num_points_corr)[0].cpu()

            # 2 step process: first downsample each downsamples GT tooth to the same number of points, then downsample the whole set
            # this is about 4 times faster than downsampling the whole set at once
            num_points_corr = torch.ones(len(self.toothlist)-1)*self.num_points_corr/(len(self.toothlist)-1)
            num_points_corr = torch.ceil(num_points_corr).int().to(self.device)
            corr_tensor_sampled = fps(corr_tensor, K=num_points_corr)[0].view(1,-1,3)
            corr_tensor_sampled = fps(corr_tensor_sampled, K=self.num_points_corr)[0]

            corr = torch.cat((corr, corr_tensor_sampled), dim=0)

        if corr_tooth is not None:
            return gt.cpu()[corr_tooth_idx], corr.cpu().squeeze(0)
        else:
            return gt.cpu(), corr.cpu()
        
            
    def __getitem__(self, idx):
        # patient, tooth = self.patient_tooth_list[idx]
        # get index of first occurence of patient in self.patient_tooth_list

        if self.enable_cache:

            if not self.use_cached_data:
                patient = self.patientlist[idx]
                first_occurence = next((i for i, x in enumerate(self.patient_tooth_list) if x[0] == patient), None)

                if patient not in self.loaded_patients:
                    print(f'Loading patient {patient}...')
                    gt, corr = self.load_patient_data(patient)
                    self.loaded_patients.append(patient)

                    self.shared_array_gt[first_occurence:first_occurence+gt.shape[0]] = gt
                    self.shared_array_corr[first_occurence:first_occurence+corr.shape[0]] = corr

            if sorted(self.loaded_patients) == sorted(self.patientlist):
                self.set_use_cached_data(True)

            if self.use_cached_data:
                # save the shared_arrays to /storage/share/temp
                if (not op.exists(self.cache_path) and self.create_cache_file) or (op.exists(self.cache_path) and self.overwrite_cache_file):
                    os.makedirs(self.cache_dir, exist_ok=True)
                    np.savez_compressed(self.cache_path, shared_array_gt=self.shared_array_gt.numpy(), shared_array_corr=self.shared_array_corr.numpy())
                    self.create_cache_file = False

            return self.shared_array_gt[idx], self.shared_array_corr[idx]
        
        else:
            patient, tooth = self.patient_tooth_list[idx]
            return self.load_patient_data(patient, tooth)


if __name__ == '__main__':
    # initialize all cuda gpus
    # print(torch.cuda.is_available())
    device = torch.device('cuda:1')

    from TeethSegDataset_onlyvertices import TeethSegDataset as TeethSegDataset_onlyvertices

    train_set = TeethSegDataset(
        mode='train',
        num_points_corr=8192,
        num_points_gt=4096,
        device=device,
        splits={'train': 0.02, 'val': 0},
        enable_cache=True,
        create_cache_file=True
    )

    train_loader = torch.utils.data.DataLoader(
            train_set, 
            batch_size=16,
            shuffle=False, 
            pin_memory=False,
            num_workers=0)

    # # load val and train into cache if specified
    # if train_loader.dataset.enable_cache:
    #     print(f'[Train] Filling cache...')
    #     t0 = time.time()
    #     for i, data in enumerate(train_loader):
    #         pass

    #     print(f'[Train] Filling cache took {format_duration(time.time()-t0)}',)

    for epoch in range(1000):
        t0 = time.time()
        for i, data in enumerate(train_loader):
            gt = data[0].to(device)
            corr = data[1].to(device)
            # print(data[0].shape, data[1].shape)
            pass
        print(f'Epoch {epoch} took {format_duration(time.time()-t0)}')

    # import open3d as o3d

    #         # print(data[0].shape)
    #         # print(data[0].__class__)
    #         # visualize data
    #         # pcd = o3d.geometry.PointCloud()
    #         # pcd.points = o3d.utility.Vector3dVector(data[1][0])
    #         # o3d.visualization.draw_geometries([pcd])

    #     
