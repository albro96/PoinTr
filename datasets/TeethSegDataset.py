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

import copy
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

class TeethSegDataset(Dataset):
    def __init__(
            self, 
            mode='train', 
            tooth_range=None, 
            num_points_gt=1024, 
            num_points_corr=1024, 
            num_points_corr_type='full', # 'full' or 'single
            num_points_orig=8192,
            gt_type='single', 
            data_type='npy',
            samplingmethod='fps',
            downsample_steps=2,
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
        self.samplingmethod = samplingmethod
        self.downsample_steps = downsample_steps
        self.data_filter_path = pada.datasets.TeethSeg22.data_filter_path
        self.toothlabels_path = pada.datasets.TeethSeg22.toothlabels_path
        self.device = device if device is not None else torch.device('cpu')
        
        if self.device.type == 'cpu':
            print('Using CPU. This will slow down the data loading process. Consider using CUDA.')

        # USER INPUT
        self.tooth_range = copy.deepcopy(tooth_range) if tooth_range is not None else {'corr': '1-7', 'gt': '1-7', 'jaw': 'lower','quadrants': 'all'}
        self.num_points_gt = num_points_gt
        assert num_points_corr_type in ['full', 'single'], "num_points_corr_type must be either 'full' or 'single'."
        self.num_points_corr_single = num_points_corr if num_points_corr_type == 'single' else None
        self.num_points_corr = num_points_corr if num_points_corr_type == 'full' else None
        self.num_points_orig = num_points_orig
        self.gt_type = gt_type
        assert self.gt_type in ['single', 'full'], "Single tooth or fullband ground truth is supported."
        self.mode = mode
        self.splits = {'train': 0.8,'val': 0.1, 'test': 0.1} if splits is None else splits

        if 'test' not in self.splits.keys():
            self.splits['test'] = 1 - sum(self.splits.values())

        self.enable_cache = enable_cache      

        if not self.enable_cache:
            print('Caching is disabled. This will slow down the data loading process.')

        self.quad_dict = dict(lower=[3,4], upper=[1,2])

        if self.tooth_range['quadrants'] == 'all':
            self.tooth_range['quadrants'] = self.quad_dict[self.tooth_range['jaw']]


        self.toothlist = {}
        keys = ['corr', 'gt']

        for key in keys:

            range_type = key
            tooth_range = self.tooth_range[range_type]

            if range_type == 'quadrants':
                continue

            if tooth_range != 'full':
                if isinstance(tooth_range, str):
                    start = int(tooth_range.split('-')[0])
                    end = int(tooth_range.split('-')[1])
                    self.tooth_range[range_type] = [start, end]
                    self.toothlist[range_type] = []

                    assert (start < 10)==(end < 10), "Tooth Range must be either as e.g. ."

                    if start < 10:
                        for quadrant in self.tooth_range['quadrants']:
                            assert quadrant in self.quad_dict[self.tooth_range['jaw']]
                            self.toothlist[range_type].extend([int(f'{quadrant}{tooth}') for tooth in range(self.tooth_range[range_type][0], self.tooth_range[range_type][-1]+1)])
                    else:
                        self.toothlist[range_type].extend([tooth for tooth in range(self.tooth_range[range_type][0], self.tooth_range[range_type][-1]+1)])
                else:
                    self.toothlist[range_type] = []
                    tooth_range_init = copy.deepcopy(tooth_range)
                    # check if all elements are smaller than 10
                    for tooth in tooth_range_init:
                        
                        if tooth < 10:
                            for quadrant in self.tooth_range['quadrants']:
                                assert quadrant in self.quad_dict[self.tooth_range['jaw']]
                                self.toothlist[range_type].append(int(f'{quadrant}{tooth}'))
                        else:
                            self.toothlist[range_type].append(tooth)
                        
                    # make sure that self.toothlist[range_type] is sorted and unique
                    self.toothlist[range_type] = sorted(list(set(self.toothlist[range_type])))

        # print(self.toothlist)
        print('Current toothlist:', end=' ')
        pprint(self.toothlist)

        if self.num_points_corr_single is not None:
            self.num_points_corr = self.num_points_corr_single*(len(self.toothlist['corr'])-1) 
        print(f'Number of points for corr: {self.num_points_corr}')
        print(f'Number of points for gt: {self.num_points_gt}')

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

        self.filterlist = sorted([patient for patient in self.data_filter['patient-filter'] if all(int(elem) in self.toothdict[patient] for elem in self.toothlist['corr'])])

        assert np.sum([i for i in self.splits.values()]) == 1, "The sum of split ratios must be equal to 1."

        self.patient_tooth_list = []
        for patient in self.filterlist:
            for tooth in self.toothlist['gt']:
                self.patient_tooth_list.append([patient, tooth])

        self.num_samples = len(self.patient_tooth_list)
        print(f'Number of samples [total]: {self.num_samples}')

        # Calculate the indices where to split the array
        splitnums = {'train': 0, 'val': 1, 'test': 2}

        train_index = int(self.num_samples*self.splits['train'])
        val_index = train_index + int(self.num_samples*self.splits['val'])

        patientlist_split = np.split(np.array(self.patient_tooth_list), [train_index, val_index])
        self.patient_tooth_list = patientlist_split[splitnums[self.mode]].tolist()
        print(f'Number of samples [{self.mode}]: {len(self.patient_tooth_list)}')
    
        if self.enable_cache:
            cache_len = self.num_samples

            # create hash for cache
            relevant_keys = ['toothlist', 'num_points_gt', 'num_points_corr', 'num_points_orig', 'gt_type']

            cache_dict = {key: value for key, value in self.__dict__.items() if key in relevant_keys}

            cache_dict['filterlist'] = sorted(self.filterlist)

            self.cache_hash = sha256(json.dumps(cache_dict, sort_keys=True).encode()).hexdigest()[:8]
            print(f'Current data-cache hash: {self.cache_hash}')
            self.cache_path_gt = op.join(self.cache_dir, f'{self.cache_hash}-gt.pt')
            self.cache_path_corr = op.join(self.cache_dir, f'{self.cache_hash}-corr.pt')

            shared_array_base_gt = mp.Array(ctypes.c_float, cache_len*self.num_points_gt*3)
            shared_array_gt = np.ctypeslib.as_array(shared_array_base_gt.get_obj())
            shared_array_gt = shared_array_gt.reshape(cache_len, self.num_points_gt, 3)
            self.shared_array_gt = torch.from_numpy(shared_array_gt)

            shared_array_base_corr = mp.Array(ctypes.c_float, cache_len*self.num_points_corr*3)
            shared_array_corr = np.ctypeslib.as_array(shared_array_base_corr.get_obj())
            shared_array_corr = shared_array_corr.reshape(cache_len, self.num_points_corr, 3)
            self.shared_array_corr = torch.from_numpy(shared_array_corr)

            self.load_cache()

            files_exist = op.exists(self.cache_path_corr) and op.exists(self.cache_path_gt)
            if not files_exist and self.create_cache_file:
                print('Cache files do not exist. Creating cache...')
                self.save_cache()
            elif files_exist and self.overwrite_cache_file:
                print('Overwriting cache files...')
                self.save_cache()

            # Split the array
            gt_split = np.split(self.shared_array_gt, [train_index, val_index])
            corr_split = np.split(self.shared_array_corr, [train_index, val_index])

            self.shared_array_corr = corr_split[splitnums[self.mode]]
            self.shared_array_gt = gt_split[splitnums[self.mode]]
        
     

    def __len__(self):
        return len(self.patient_tooth_list)

    def downsample_batched_pcd(self, pcd, num_points, samplingmethod=None, steps=2):
        '''
        Downsamples the input data tensor to the specified number of points using the specified sampling method.
        Args:
            pcd (torch.Tensor): The input data tensor of shape (batch_size, num_points, 3).
            num_points (int): The desired number of points after downsampling.
            samplingmethod (str, optional): The sampling method to use. Defaults to 'fps'.
            steps (int, optional): The number of downsampling steps to perform. Only 1 or 2 steps are supported. Defaults to 2.
        Returns:
            torch.Tensor: The downsampled data tensor of shape (batch_size, num_points, 3).
        '''

        if samplingmethod is None:
            samplingmethod = 'fps'

        assert steps in [1,2], "Only 1 or 2 steps are supported."

        if steps == 2:
            # first downsample each item to the same number of points, then downsample the whole batch
            # this is about 4 times faster than downsampling the whole set at once

            batchsize = pcd.shape[0]

            num_points_tmp = torch.ones(batchsize)*num_points/batchsize
            num_points_tmp = torch.ceil(num_points_tmp).int().to(self.device)

            if samplingmethod == 'fps':
                pcd = fps(pcd, K=num_points_tmp)[0].view(1,-1,3)

        if samplingmethod == 'fps':
            pcd = fps(pcd, K=num_points)[0]
        
        return pcd
        

    def load_patient_data(self, patient, corr_tooth=None):
        # create paths
        all_teeth_tensor = torch.empty(0, self.num_points_orig, 3)
        # start = time.time()
        for tooth in self.toothlist['corr']:
            pcd_path = op.join(self.data_dir, f'{self.data_type}_{self.num_points_orig}',patient, f"{tooth}.{self.data_type}")
            tooth_tensor = torch.from_numpy(np.load(pcd_path)).unsqueeze(0)
            all_teeth_tensor = torch.cat((all_teeth_tensor, tooth_tensor), dim=0)

        # print(f'Loading patient data took {time.time()-start}')

        all_teeth_tensor = all_teeth_tensor.to(self.device)

        # start = time.time()
        gt = fps(all_teeth_tensor, K=self.num_points_gt)[0]

        corr = torch.empty(0, self.num_points_corr, 3).to(self.device)

        filter_arr = np.array(self.toothlist['corr']) 
        filter_arr_gt = np.array(self.toothlist['gt'])
        # get indices of filter_arr that are in filter_arr_gt
        indices_gt = np.where(np.isin(filter_arr, filter_arr_gt))[0]

        for tooth in self.toothlist['gt']:
            if corr_tooth is not None:
                if tooth != corr_tooth:
                    continue
            
            corr_tensor = gt[filter_arr != tooth]

            if self.num_points_corr_single is not None:
                # resample the single teeth from the sampled gts to the desired number of points and concat on the first dimension
                corr_tensor_sampled = fps(corr_tensor, K=self.num_points_corr_single)[0].view(1,-1,3)
            else:
                corr_tensor_sampled = self.downsample_batched_pcd(pcd=corr_tensor, num_points=self.num_points_corr, samplingmethod=self.samplingmethod, steps=self.downsample_steps)

            corr = torch.cat((corr, corr_tensor_sampled), dim=0)

        if self.gt_type == 'full':
            gt = self.downsample_batched_pcd(pcd=gt, num_points=self.num_points_gt, samplingmethod=self.samplingmethod, steps=self.downsample_steps)
            gt = gt.repeat(corr.shape[0], 1, 1)

        corr = corr.cpu()  
        gt = gt.cpu()

        if self.gt_type == 'single':
            if corr_tooth is not None: 
                return corr.squeeze(0), gt[filter_arr == corr_tooth].squeeze(0)
            else: 
                return corr, gt[indices_gt]
        else: 
            if corr_tooth is not None: 
                return corr.squeeze(0), gt.squeeze(0) 
            else:
                return corr, gt, 
        
    def load_cache(self):
        # load cache     
        if (op.exists(self.cache_path_gt) and not self.overwrite_cache_file) and\
            (op.exists(self.cache_path_corr) and not self.overwrite_cache_file):
            print(f'Loading GT cache:\t{self.cache_path_gt}')
            self.shared_array_gt = torch.load(self.cache_path_gt)
            print(f'Loading CORR cache:\t{self.cache_path_corr}')
            self.shared_array_corr = torch.load(self.cache_path_corr)
        else:
            print('Loading data for cache ...')
            for idx, patient in enumerate(tqdm(self.filterlist)):
                idx_ref = len(self.toothlist['gt'])
                start = idx*idx_ref 
                end = start+idx_ref

                corr, gt = self.load_patient_data(patient)
                self.shared_array_gt[start:end] = gt
                self.shared_array_corr[start:end] = corr        

    def save_cache(self):
        print(f'Saving cache to: \nGT:\t{self.cache_path_gt}\nCORR:\t{self.cache_path_corr}')
        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(self.shared_array_gt, self.cache_path_gt)
        torch.save(self.shared_array_corr, self.cache_path_corr)
            
    def __getitem__(self, idx):
        patient, tooth = self.patient_tooth_list[idx]
        if self.enable_cache:
            return self.shared_array_corr[idx], self.shared_array_gt[idx]
        else:
            return self.load_patient_data(patient, corr_tooth=int(tooth))
            