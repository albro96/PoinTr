import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *

# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py


@DATASETS.register_module()
class TeethSeg(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config, subset=None):
        # self.partial_points_path = config.PARTIAL_POINTS_PATH
        # self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.category_dict = config.CATEGORY_DICT
        self.npoints_gt = config.N_POINTS_GT
        self.npoints_partial = config.N_POINTS_PARTIAL
        # self.npoints = config.N_POINTS
        self.gt_type = config.GT_TYPE
        self.data_dir = config.DATA_DIR
        self.dataformat = config.DATAFORMAT
        self.corr_type = config.CORR_TYPE

        if subset is None:
            self.subset = config.subset
        else:
            self.subset = subset

        # check if all value lists in category_dict are the same length
        num_renderings = len(self.category_dict[list(self.category_dict.keys())[0]])
        for key in self.category_dict.keys():
            assert len(self.category_dict[key]) == num_renderings

        # Load the dataset indexing file
        self.dataset_categories = []

        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())

        # crop the dataset_categories to only include the categories in category_dict
        self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_name'] in self.category_dict.keys()]

        print([dc['taxonomy_name'] for dc in self.dataset_categories])

        self.n_renderings = num_renderings if self.subset == 'train' else 1
        
        # self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.file_list = self._get_file_list(self.subset)

        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose(
                [
                    {
                    'callback': 'RandomSamplePoints',
                    'parameters': {'n_points': 2048},
                    'objects': ['partial']
                    }, 
                    {
                    'callback': 'RandomMirrorPoints',
                    'objects': ['partial', 'gt']
                    },
                    {
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                    }
                ]
            )
        else:
            return data_transforms.Compose(
                [
                    {
                    'callback': 'RandomSamplePoints',
                    'parameters': {'n_points': 2048},
                    'objects': ['partial']
                    }, 
                    {
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                    }
                ]
            )

    # def _get_file_list(self, subset, n_renderings=1):
    def _get_file_list(self, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
            samples = dc[subset]

            jawtype = dc['taxonomy_name'].split('_')[0]
            toothrange = dc['taxonomy_name'].split('_')[1]

            if self.gt_type == 'full':
                # full GT
                for s in samples:
                    
                    file_list.append({
                        'taxonomy_id':  dc['taxonomy_id'],
                        'model_id':     s,
                        # 'partial_path': [self.partial_points_path % ('/'.join(dc['taxonomy_name'].split('_')), self.npoints_partial, s, tooth) for tooth in self.category_dict[dc['taxonomy_name']]],
                        # 'gt_path':      self.complete_points_path % ('/'.join(dc['taxonomy_name'].split('_')), self.npoints_gt, s),
                        'partial_path': [os.path.join(self.data_dir, self.corr_type, jawtype, toothrange, str(self.npoints_partial), s, tooth + f'.{self.dataformat}') for tooth in self.category_dict[dc['taxonomy_name']]],
                        'gt_path':      os.path.join(self.data_dir, 'full', jawtype, toothrange, str(self.npoints_gt), s + f'.{self.dataformat}'),

                    })
            elif self.gt_type == 'single':
                # single GT
                for s in samples:
                    for tooth in self.category_dict[dc['taxonomy_name']]:
                        file_list.append({
                            'taxonomy_id':  dc['taxonomy_id'],
                            'model_id':     s,
                            # 'partial_path': self.partial_points_path % ('/'.join(dc['taxonomy_name'].split('_')), self.npoints_partial, s, tooth),
                            # 'gt_path': self.complete_points_path % (self.npoints_gt, s, tooth),
                            'partial_path': os.path.join(self.data_dir, self.corr_type, jawtype, toothrange, str(self.npoints_partial), s, tooth + f'.{self.dataformat}'),
                            'gt_path': os.path.join(self.data_dir, 'gt', str(self.npoints_gt), s, tooth + f'.{self.dataformat}'),

                        })
            else:
                raise ValueError('Invalid gt_type: %s' % self.gt_type)

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        assert data['gt'].shape[0] == self.npoints_gt

        if self.transforms is not None:
            data = self.transforms(data)

        # print("------------------SHAPES------------------")
        # print(data['partial'].shape, data['gt'].shape)

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)
