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
    def __init__(self, mode='train', jaw='lower', quadrants='all', tooth_range='1-7', num_points_gt=16, num_points_corr=512, gt_type='single', splits=None, cache_data=True, device='cuda'):
        # PATHS
        pada = import_dir_path(filepath=__file__)
        self.data_dir = pada.datasets.TeethSeg22.data_npy_dir
        self.data_filter_path = pada.datasets.TeethSeg22.data_filter_path
        self.toothlabels_path = pada.datasets.TeethSeg22.toothlabels_path
        self.device = device

        # USER INPUT
        self.jaw = jaw
        self.quadrants = quadrants
        self.tooth_range = tooth_range
        self.num_points_gt = num_points_gt
        self.num_points_corr = num_points_corr
        self.gt_type = gt_type
        self.mode = mode
        self.splits = {'train': 0.85,'val': 0.1} if splits is None else splits
        self.cache_data = cache_data

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

        print(len(self.patientlist))

        self.loaded_patients = []

        self.patient_tooth_list = []
        for patient in self.patientlist:
            for tooth in self.toothlist:
                self.patient_tooth_list.append([patient, tooth])
        
        self.num_samples = len(self.patient_tooth_list)

        if self.cache_data:
            shared_array_base_gt = mp.Array(ctypes.c_float, self.num_samples*self.num_points_gt*3)
            shared_array_gt = np.ctypeslib.as_array(shared_array_base_gt.get_obj())
            shared_array_gt = shared_array_gt.reshape(self.num_samples, self.num_points_gt, 3)
            self.shared_array_gt = torch.from_numpy(shared_array_gt)

            shared_array_base_corr = mp.Array(ctypes.c_float, self.num_samples*self.num_points_corr*3)
            shared_array_corr = np.ctypeslib.as_array(shared_array_base_corr.get_obj())
            shared_array_corr = shared_array_corr.reshape(self.num_samples, self.num_points_corr, 3)
            self.shared_array_corr = torch.from_numpy(shared_array_corr)

        self.use_cache = False

    def set_use_cache(self, use_cache):
            self.use_cache = use_cache

    def __len__(self):
        if self.use_cache:
            return len(self.patient_tooth_list)
        else:
            return len(self.patientlist)


    def load_patient_data(self, patient):
        # create paths
        mesh_path = op.join(self.data_dir, patient, f"{patient}_{self.jaw}.npy")
        labels_path = op.join(self.data_dir, patient, f"{patient}_{self.jaw}.json")

        # load labels as dict
        with open(labels_path, 'r') as f:
            labels = EasyDict(json.load(f))

        # define gingiva_label
        gingiva_label = 0

        # get a list of all unique labels from labels.labels and drop all zeros
        label_list = np.unique(labels.labels)
        label_list = label_list[label_list!=gingiva_label].tolist()

        mesh_vertices = np.load(mesh_path)

        self.vertices = {patient: {'gt': {}, 'corr': {}}}

        gt = None
        corr = None

        for tooth in self.toothlist:
            # get arr with same length as vertices_arr that contains the label for each vertex
            labels_arr = np.asarray(labels.labels)
        
            label_filter_gt = labels_arr==tooth
            # self.vertices[patient]['gt'][tooth] = mesh_vertices[label_filter_gt]

            mesh_vertices_filtered_gt = torch.from_numpy(mesh_vertices[label_filter_gt]).float().unsqueeze(0).to(self.device)
            mesh_vertices_filtered_gt = fps(mesh_vertices_filtered_gt, K=self.num_points_gt)[0].cpu()

            # mesh_vertices_filtered_gt = torch.from_numpy(mesh_vertices[label_filter_gt]).float().unsqueeze(0)
            # mesh_vertices_filtered_gt = fps(mesh_vertices_filtered_gt, K=self.num_points_gt)[0]

            if gt is None:
                gt = mesh_vertices_filtered_gt
            else:
                gt = torch.cat((gt, mesh_vertices_filtered_gt), 0)

            # create new toothlist without label
            toothlist_corr = self.toothlist.copy()
            toothlist_corr.remove(tooth)

            label_filter_corr = np.isin(labels_arr, toothlist_corr)
            mesh_vertices_filtered_corr = torch.from_numpy(mesh_vertices[label_filter_corr]).float().unsqueeze(0).to(self.device)
            mesh_vertices_filtered_corr = fps(mesh_vertices_filtered_corr, K=self.num_points_corr)[0].cpu()

            # mesh_vertices_filtered_corr = torch.from_numpy(mesh_vertices[label_filter_corr]).float().unsqueeze(0)
            # mesh_vertices_filtered_corr = fps(mesh_vertices_filtered_corr, K=self.num_points_corr)[0]

            if corr is None:
                corr = mesh_vertices_filtered_corr
            else:
                corr = torch.cat((corr, mesh_vertices_filtered_corr), 0)
        
        return gt, corr

    # def loadcache(self, idx):
    #     patient = self.patientlist[idx]
    #     first_occurence = next((i for i, x in enumerate(self.patient_tooth_list) if x[0] == patient), None)

    #     print(f'Loading patient {patient}...')
    #     gt, corr = self.load_patient_data(patient)
    #     self.loaded_patients.append(patient)

    #     self.shared_array_gt[first_occurence:first_occurence+len(gt)] = gt
    #     self.shared_array_corr[first_occurence:first_occurence+len(corr)] = corr


    def __getitem__(self, idx):
        # patient, tooth = self.patient_tooth_list[idx]

        # get index of first occurence of patient in self.patient_tooth_list

        if self.cache_data:
            if not self.use_cache:
                patient = self.patientlist[idx]
                first_occurence = next((i for i, x in enumerate(self.patient_tooth_list) if x[0] == patient), None)

                if patient not in self.loaded_patients:
                    print(f'Loading patient {patient}...')
                    gt, corr = self.load_patient_data(patient)
                    self.loaded_patients.append(patient)

                    self.shared_array_gt[first_occurence:first_occurence+len(gt)] = gt
                    self.shared_array_corr[first_occurence:first_occurence+len(corr)] = corr

            return self.shared_array_gt[idx], self.shared_array_corr[idx]


device = torch.device('cuda')

train_set = TeethSegDataset(mode='train', num_points_corr=4096, num_points_gt=1024, device=device, splits={'train': 0.02,'val': 0.14}, cache_data=True)

# print(train_set[0])

train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=1,
        shuffle=False, 
        pin_memory=False,
        num_workers=24)

# load val and train into cache if specified
if train_loader.dataset.cache_data and not train_loader.dataset.use_cache:
    print(f'[Train] Filling cache...')
    t0 = time.time()
    for i, data in enumerate(train_loader):
        pass
    print(f'[Train] Filling cache took {format_duration(time.time()-t0)}',)

import open3d as o3d
for epoch in range(5):
    if epoch==0:
        train_loader.dataset.set_use_cache(True)

    print(len(train_loader))

    t0 = time.time()
    for i, data in enumerate(train_loader): 
        # visualize data
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[1][0].cpu().numpy())
        o3d.visualization.draw_geometries([pcd])



    print(f'[Train] {time.time()-t0}',)

        



# myarr = np.arange(10)
# print(myarr)

# myarr2 = np.arange(3)

# myarr[3:6] = myarr2
# print(myarr)

# # PATHS
# pada = import_dir_path(filepath=__file__)
# data_dir = pada.datasets.TeethSeg22.data_npy_dir
# data_filter_path = pada.datasets.TeethSeg22.data_filter_path
# toothlabels_path = pada.datasets.TeethSeg22.toothlabels_path

# # ----------------------------------------------------- #

# # USER INPUT
# dry_run = False
# visu = False
# concat = True

# jaw = 'lower'
# quadrants = 'all' # [3,4] #
# tooth_range = '1-7' #['full', [1,2], [1,3], [1,5], [1,6], [1,7], [1,8], [6,8], [5,7]]

# num_points_gt = 2028
# num_points_corr = 8192

# # ----------------------------------------------------- #
# gt_type = 'single'

# quad_dict = dict(
#                 lower=[3,4], 
#                 upper=[1,2]
#                 )

# if quadrants == 'all':
#     quadrants = quad_dict[jaw]

# if tooth_range != 'full':
#     if isinstance(tooth_range, str):
#         tooth_range_list = [int(tooth_range[0]), int(tooth_range[-1])]

# toothlist = []
# for quadrant in quadrants:
#     assert quadrant in quad_dict[jaw]
#     toothlist.extend([int(f'{quadrant}{tooth}') for tooth in range(tooth_range_list[0], tooth_range_list[-1]+1)])

# # create data_filter list if it does not exist
# if not op.exists(data_filter_path):
#     print('Creating new data filter list..')
#     data_filter = pd.DataFrame()
#     data_filter['patient-all'] = os.listdir(data_dir)
#     data_filter['patient-filter'] = data_filter.loc[:, 'patient-all']
#     data_filter.to_csv(data_filter_path, sep=';', index=False)


# # load data_filter list
# data_filter = pd.read_csv(filepath_or_buffer=data_filter_path, sep=';', usecols=['patient-filter']).dropna().reset_index(drop=True)

# # load the toothdict
# with open(toothlabels_path, 'r') as f:
#     toothdict = EasyDict(json.load(f))

# mode = 'test'
# splits = {'train': 0.8,'val': 0.1,'test': 0.1}

# filterlist = [patient for patient in data_filter['patient-filter'] if all(int(elem) in toothdict[patient] for elem in toothlist)]

# # Calculate the indices where to split the array
# train_index = int(len(filterlist)*splits['train'])
# val_index = train_index + int(len(filterlist)*splits['val'])

# # Split the array
# train, val, test = np.split(filterlist, [train_index, val_index])

# # Assign to patientlist based on mode
# if mode == 'train':
#     patientlist = train
# elif mode == 'val':
#     patientlist = val
# elif mode == 'test':
#     patientlist = test

# patient_tooth_list = []
# for patient in tqdm(patientlist):
#     for tooth in toothlist:
#         patient_tooth_list.append([patient, tooth])

# mesh_vertices = {}
# labels = {}
# ctr = 0
# # loop over all patients and jawtypes
# for idx, patienttooth in enumerate(tqdm(patient_tooth_list)):
#     patient = patienttooth[0]
#     tooth = patienttooth[1]

#     if patient != patient_tooth_list[idx-1][0]:
#         ctr += 1
#         # create paths
#         mesh_path = op.join(data_dir, patient, f"{patient}_{jaw}.npy")
#         labels_path = op.join(data_dir, patient, f"{patient}_{jaw}.json")

#         # load labels as dict
#         with open(labels_path, 'r') as f:
#             labels = EasyDict(json.load(f))

#         # define gingiva_label
#         gingiva_label = 0

#         # get a list of all unique labels from labels.labels and drop all zeros
#         label_list = np.unique(labels.labels)
#         label_list = label_list[label_list!=gingiva_label].tolist()

#         mesh_vertices = np.load(mesh_path)

#         vertices = {'gt': {}, 'corr': {}}
#         for tooth in toothlist:
#             # get arr with same length as vertices_arr that contains the label for each vertex
#             labels_arr = np.asarray(labels.labels)
        
#             label_filter_gt = labels_arr==tooth
#             vertices['gt'][tooth] = mesh_vertices[label_filter_gt]

#             # create new toothlist without label
#             toothlist_corr = toothlist.copy()
#             toothlist_corr.remove(tooth)

#             label_filter_corr = np.isin(labels_arr, toothlist_corr)
#             vertices['corr'][tooth] = mesh_vertices[label_filter_corr]

#     else:
#         vertices_gt = vertices['gt'][tooth]
#         vertices_corr = vertices['corr'][tooth]

# # loop over all patients and jawtypes
# for patient in tqdm(patientlist):
#     # create paths
#     mesh_path = op.join(data_dir, patient, f"{patient}_{jaw}.npy")
#     labels_path = op.join(data_dir, patient, f"{patient}_{jaw}.json")

#     # load labels as dict
#     with open(labels_path, 'r') as f:
#         labels = EasyDict(json.load(f))

#     # define gingiva_label
#     gingiva_label = 0

#     # get a list of all unique labels from labels.labels and drop all zeros
#     label_list = np.unique(labels.labels)
#     label_list = label_list[label_list!=gingiva_label].tolist()

#     mesh_vertices = np.load(mesh_path)

#     # get arr with same length as vertices_arr that contains the label for each vertex
#     labels_arr = np.asarray(labels.labels)
 
#     for tooth in toothlist:
#         label_filter_gt = labels_arr==tooth
#         vertices_gt = mesh_vertices[label_filter_gt]

#         # create new toothlist without label
#         toothlist_corr = toothlist.copy()
#         toothlist_corr.remove(tooth)

#         label_filter_corr = np.isin(labels_arr, toothlist_corr)
#         vertices_corr = mesh_vertices[label_filter_corr]


# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

# @DATASETS.register_module()
# class TeethSeg(data.Dataset):
#     # def __init__(self, data_root, subset, class_choice = None):
#     def __init__(
#             self, 
#             config, 
#             mode=None
#             ):
        
#         self.data_dir = config.DATA_DIR
#         self.mode = mode
#         self.train_split = {'train': 0.85, 'val': 0.1, 'test': 0.05},
#         self.sampling_method = config.get("SAMPLING_METHOD", 'FurthestPointSample')

#         self.category_dict = EasyDict(config.CATEGORY_DICT)

#         self.jaw = self.category_dict.jaw
#         self.toothrange = self.category_dict.toothrange
#         self.recon = self.category_dict.recon

#         self.npoints_gt = config.N_POINTS_GT
#         self.npoints_partial = config.N_POINTS_PARTIAL

#         self.gt_type = config.get("GT_TYPE", "single")
#         self.corr_type = config.get("CORR_TYPE", "corr") 

#         self.file_list = self._get_file_list(self.subset)
#         self.transforms = self._get_transforms(self.subset)

#     def _get_transforms(self, subset):
#         transform_list = [
#             {
#             'callback': self.sampling_method, #'RandomSamplePoints',
#             'parameters': {'n_points': 2048},
#             'objects': ['partial']
#             }, 
#             {
#             'callback': 'RandomMirrorPoints',
#             'objects': ['partial', 'gt']
#             },
#             {
#             'callback': 'ToTensor',
#             'objects': ['partial', 'gt']
#             }
#             ]
                        
#         if self.sampling_method == 'RandomSamplePoints':
#             if subset == 'train':
#                 return data_transforms.Compose(transform_list)
#             else:
#                 return data_transforms.Compose([transform_list[0], transform_list[-1]])    
                  
#         elif self.sampling_method == 'None':
#             return data_transforms.Compose([transform_list[-1]])      
            
#         elif self.sampling_method == 'FurthestPointSampling':
#             return data_transforms.Compose([transform_list[-1], transform_list[0]])

#     # def _get_file_list(self, subset, n_renderings=1):
#     def _get_file_list(self, subset):
#         """Prepare file list for the dataset"""
#         file_list = []

#         for dc in self.dataset_categories:
#             print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
#             self.samples = dc[subset]

#             jawtype = dc['taxonomy_name'].split('_')[0]
#             toothrange = dc['taxonomy_name'].split('_')[1]

#             if self.gt_type == 'full':
#                 # full GT
#                 for s in self.samples:
                    
#                     file_list.append({
#                         'taxonomy_id':  dc['taxonomy_id'],
#                         'model_id':     s,
#                         # 'partial_path': [self.partial_points_path % ('/'.join(dc['taxonomy_name'].split('_')), self.npoints_partial, s, tooth) for tooth in self.category_dict[dc['taxonomy_name']]],
#                         # 'gt_path':      self.complete_points_path % ('/'.join(dc['taxonomy_name'].split('_')), self.npoints_gt, s),
#                         'partial_path': [os.path.join(self.data_dir, self.corr_type, jawtype, toothrange, str(self.npoints_partial), s, tooth + f'.{self.dataformat}') for tooth in self.category_dict[dc['taxonomy_name']]],
#                         'gt_path':      os.path.join(self.data_dir, 'full', jawtype, toothrange, str(self.npoints_gt), s + f'.{self.dataformat}'),

#                     })
#             elif self.gt_type == 'single':
#                 # single GT
#                 for s in self.samples:
#                     for tooth in self.category_dict[dc['taxonomy_name']]:
#                         file_list.append({
#                             'taxonomy_id':  dc['taxonomy_id'],
#                             'model_id':     s,
#                             # 'partial_path': self.partial_points_path % ('/'.join(dc['taxonomy_name'].split('_')), self.npoints_partial, s, tooth),
#                             # 'gt_path': self.complete_points_path % (self.npoints_gt, s, tooth),
#                             'partial_path': os.path.join(self.data_dir, self.corr_type, jawtype, toothrange, str(self.npoints_partial), s, tooth + f'.{self.dataformat}'),
#                             'gt_path': os.path.join(self.data_dir, 'gt', str(self.npoints_gt), s, tooth + f'.{self.dataformat}'),

#                         })
#             else:
#                 raise ValueError('Invalid gt_type: %s' % self.gt_type)

#         print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
#         return file_list

#     def __getitem__(self, idx):
#         sample = self.file_list[idx]
#         data = {}
#         rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

#         for ri in ['partial', 'gt']:
#             file_path = sample['%s_path' % ri]
#             if type(file_path) == list:
#                 file_path = file_path[rand_idx]
#             data[ri] = IO.get(file_path).astype(np.float32)

#         assert data['gt'].shape[0] == self.npoints_gt

#         if self.transforms is not None:
#             data = self.transforms(data)

#         # print("------------------SHAPES------------------")
#         # print(data['partial'].shape, data['gt'].shape)

#         return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

#     def __len__(self):
#         return len(self.file_list)




# ------------------------------------------------------------ #
   
# @DATASETS.register_module()
# class TeethSeg(data.Dataset):
#     # def __init__(self, data_root, subset, class_choice = None):
#     def __init__(self, config, subset=None):
#         # self.partial_points_path = config.PARTIAL_POINTS_PATH
#         # self.complete_points_path = config.COMPLETE_POINTS_PATH
#         self.category_file = config.CATEGORY_FILE_PATH
#         self.category_dict = config.CATEGORY_DICT
#         self.npoints_gt = config.N_POINTS_GT
#         self.npoints_partial = config.N_POINTS_PARTIAL
#         # self.npoints = config.N_POINTS
#         self.gt_type = config.get("GT_TYPE", "single")
#         self.data_dir = config.DATA_DIR
#         self.dataformat = config.get("DATAFORMAT", "pcd") 
#         self.corr_type = config.get("CORR_TYPE", "corr") 
#         self.sampling_method = config.get("SAMPLING_METHOD", 'RandomSamplePoints')
        

#         if subset is None:
#             self.subset = config.subset
#         else:
#             self.subset = subset

#         # check if all value lists in category_dict are the same length
#         num_renderings = len(self.category_dict[list(self.category_dict.keys())[0]])
#         for key in self.category_dict.keys():
#             assert len(self.category_dict[key]) == num_renderings

#         # Load the dataset indexing file
#         self.dataset_categories = []

#         with open(self.category_file) as f:
#             self.dataset_categories = json.loads(f.read())

#         # crop the dataset_categories to only include the categories in category_dict
#         self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_name'] in self.category_dict.keys()]

#         print([dc['taxonomy_name'] for dc in self.dataset_categories])

#         self.n_renderings = num_renderings if self.subset == 'train' else 1
        
#         # self.file_list = self._get_file_list(self.subset, self.n_renderings)
#         self.file_list = self._get_file_list(self.subset)

#         self.transforms = self._get_transforms(self.subset)

#     def _get_transforms(self, subset):
#         transform_list = [
#             {
#             'callback': self.sampling_method, #'RandomSamplePoints',
#             'parameters': {'n_points': 2048},
#             'objects': ['partial']
#             }, 
#             {
#             'callback': 'RandomMirrorPoints',
#             'objects': ['partial', 'gt']
#             },
#             {
#             'callback': 'ToTensor',
#             'objects': ['partial', 'gt']
#             }
#             ]
                        
#         if self.sampling_method == 'RandomSamplePoints':
#             if subset == 'train':
#                 return data_transforms.Compose(transform_list)
#             else:
#                 return data_transforms.Compose([transform_list[0], transform_list[-1]])    
                  
#         elif self.sampling_method == 'None':
#             return data_transforms.Compose([transform_list[-1]])      
            
#         elif self.sampling_method == 'FurthestPointSampling':
#             return data_transforms.Compose([transform_list[-1], transform_list[0]])

#     # def _get_file_list(self, subset, n_renderings=1):
#     def _get_file_list(self, subset):
#         """Prepare file list for the dataset"""
#         file_list = []

#         for dc in self.dataset_categories:
#             print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
#             self.samples = dc[subset]

#             jawtype = dc['taxonomy_name'].split('_')[0]
#             toothrange = dc['taxonomy_name'].split('_')[1]

#             if self.gt_type == 'full':
#                 # full GT
#                 for s in self.samples:
                    
#                     file_list.append({
#                         'taxonomy_id':  dc['taxonomy_id'],
#                         'model_id':     s,
#                         # 'partial_path': [self.partial_points_path % ('/'.join(dc['taxonomy_name'].split('_')), self.npoints_partial, s, tooth) for tooth in self.category_dict[dc['taxonomy_name']]],
#                         # 'gt_path':      self.complete_points_path % ('/'.join(dc['taxonomy_name'].split('_')), self.npoints_gt, s),
#                         'partial_path': [os.path.join(self.data_dir, self.corr_type, jawtype, toothrange, str(self.npoints_partial), s, tooth + f'.{self.dataformat}') for tooth in self.category_dict[dc['taxonomy_name']]],
#                         'gt_path':      os.path.join(self.data_dir, 'full', jawtype, toothrange, str(self.npoints_gt), s + f'.{self.dataformat}'),

#                     })
#             elif self.gt_type == 'single':
#                 # single GT
#                 for s in self.samples:
#                     for tooth in self.category_dict[dc['taxonomy_name']]:
#                         file_list.append({
#                             'taxonomy_id':  dc['taxonomy_id'],
#                             'model_id':     s,
#                             # 'partial_path': self.partial_points_path % ('/'.join(dc['taxonomy_name'].split('_')), self.npoints_partial, s, tooth),
#                             # 'gt_path': self.complete_points_path % (self.npoints_gt, s, tooth),
#                             'partial_path': os.path.join(self.data_dir, self.corr_type, jawtype, toothrange, str(self.npoints_partial), s, tooth + f'.{self.dataformat}'),
#                             'gt_path': os.path.join(self.data_dir, 'gt', str(self.npoints_gt), s, tooth + f'.{self.dataformat}'),

#                         })
#             else:
#                 raise ValueError('Invalid gt_type: %s' % self.gt_type)

#         print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
#         return file_list

#     def __getitem__(self, idx):
#         sample = self.file_list[idx]
#         data = {}
#         rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

#         for ri in ['partial', 'gt']:
#             file_path = sample['%s_path' % ri]
#             if type(file_path) == list:
#                 file_path = file_path[rand_idx]
#             data[ri] = IO.get(file_path).astype(np.float32)

#         assert data['gt'].shape[0] == self.npoints_gt

#         if self.transforms is not None:
#             data = self.transforms(data)

#         # print("------------------SHAPES------------------")
#         # print(data['partial'].shape, data['gt'].shape)

#         return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

#     def __len__(self):
#         return len(self.file_list)
