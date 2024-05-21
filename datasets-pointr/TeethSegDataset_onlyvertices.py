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
from pytorch3d.structures import Pointclouds, join_pointclouds_as_batch
from pytorch3d.ops import convert_pointclouds_to_tensor


class TeethSegDataset(Dataset):
    def __init__(
            self, 
            mode='train', 
            jaw='lower', 
            quadrants='all', 
            tooth_range='1-7', 
            num_points_gt=1024, 
            num_points_corr=1024, 
            gt_type='single', 
            splits=None, 
            enable_cache=True, 
            device=None
            ):
        # PATHS
        pada = import_dir_path(filepath=__file__)
        self.data_dir = pada.datasets.TeethSeg22.data_npy_dir
        self.data_filter_path = pada.datasets.TeethSeg22.data_filter_path
        self.toothlabels_path = pada.datasets.TeethSeg22.toothlabels_path
        self.device = device if device is not None else torch.device('cpu')

        # USER INPUT
        self.jaw = jaw
        self.quadrants = quadrants
        self.tooth_range = tooth_range
        self.num_points_gt = num_points_gt
        self.num_points_corr = num_points_corr
        self.gt_type = gt_type
        self.mode = mode
        self.splits = {'train': 0.8,'val': 0.1, 'test': 0.1} if splits is None else splits

        if 'test' not in self.splits.keys():
            self.splits['test'] = 1 - sum(self.splits.values())

        self.enable_cache = enable_cache

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

        if self.enable_cache:
            shared_array_base_gt = mp.Array(ctypes.c_float, self.num_samples*self.num_points_gt*3)
            shared_array_gt = np.ctypeslib.as_array(shared_array_base_gt.get_obj())
            shared_array_gt = shared_array_gt.reshape(self.num_samples, self.num_points_gt, 3)
            self.shared_array_gt = torch.from_numpy(shared_array_gt)

            shared_array_base_corr = mp.Array(ctypes.c_float, self.num_samples*self.num_points_corr*3)
            shared_array_corr = np.ctypeslib.as_array(shared_array_base_corr.get_obj())
            shared_array_corr = shared_array_corr.reshape(self.num_samples, self.num_points_corr, 3)
            self.shared_array_corr = torch.from_numpy(shared_array_corr)

            num_teeth = mp.Array(ctypes.c_int, self.num_samples)
            self.num_teeth = np.ctypeslib.as_array(num_teeth.get_obj())


        self.use_cached_data = False

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
    
    def load_patient_data(self, patient, corr_toothlist):

        corr_toothlist = corr_toothlist if corr_toothlist is not None else self.toothlist

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
        # Convert labels.labels to numpy array outside the loop
        labels_arr = np.asarray(labels.labels)

        mesh_vertices = np.load(mesh_path)

        gt = []
        corr = []
        gt_lens = []
        corr_lens = []

        for tooth in corr_toothlist:
            
            # get arr with same length as vertices_arr that contains the label for each vertex
            label_filter_gt = labels_arr==tooth

            mesh_vertices_filtered_gt = mesh_vertices[label_filter_gt]
            gt_lens.append(mesh_vertices_filtered_gt.shape[0])

            # get index of first zero entry in self.num_teeth
            first_occurence = next((i for i, x in enumerate(self.num_teeth) if x == 0), None)
            self.num_teeth[first_occurence] = gt_lens[-1]

            mesh_vertices_filtered_gt = Pointclouds(torch.from_numpy(mesh_vertices_filtered_gt).float().unsqueeze(0)).to(self.device) 
            gt.append(mesh_vertices_filtered_gt) 

            # create new toothlist without label
            toothlist_corr = [t for t in self.toothlist if t != tooth]

            label_filter_corr = np.isin(labels_arr, toothlist_corr)
            mesh_vertices_filtered_corr = mesh_vertices[label_filter_corr]
            corr_lens.append(mesh_vertices_filtered_corr.shape[0])
            mesh_vertices_filtered_corr = Pointclouds(torch.from_numpy(mesh_vertices_filtered_corr).float().unsqueeze(0)).to(self.device)

            corr.append(mesh_vertices_filtered_corr)

        gt = join_pointclouds_as_batch(gt).points_padded()
        corr = join_pointclouds_as_batch(corr).points_padded()
        gt_lens = torch.tensor(gt_lens).to(self.device)
        corr_lens = torch.tensor(corr_lens).to(self.device)

        # use fps on gt
        gt = fps(gt, lengths=gt_lens, K=self.num_points_gt)[0].cpu()
        corr = fps(corr, lengths=corr_lens, K=self.num_points_corr)[0].cpu()

        return gt, corr

    def __getitem__(self, idx):
        # patient, tooth = self.patient_tooth_list[idx]
        # get index of first occurence of patient in self.patient_tooth_list
        if self.enable_cache:
            if not self.use_cached_data:
                patient = self.patientlist[idx]
                first_occurence = next((i for i, x in enumerate(self.patient_tooth_list) if x[0] == patient), None)

                if patient not in self.loaded_patients:
                    print(f'Loading patient {patient}...')
                    gt, corr = self.load_patient_data(patient, self.toothlist)
                    self.loaded_patients.append(patient)

                    self.shared_array_gt[first_occurence:first_occurence+len(gt)] = gt
                    self.shared_array_corr[first_occurence:first_occurence+len(corr)] = corr

            return self.shared_array_gt[idx], self.shared_array_corr[idx]
        else:
            patient, tooth = self.patient_tooth_list[idx]
            return self.load_patient_data(patient, [tooth])


if __name__ == '__main__':
    # initialize all cuda gpus
    device = torch.device('cuda:0')


    train_set = TeethSegDataset(
        mode='train',
        num_points_corr=16,
        num_points_gt=16,
        device=device,
        splits={'train': 1, 'val': 0},
        enable_cache=True
    )

    print(len(train_set))
    # print(train_set[0][1].shape)

    train_loader = torch.utils.data.DataLoader(
            train_set, 
            batch_size=16,
            shuffle=False, 
            pin_memory=False,
            num_workers=24)

    # for data in train_loader:
    #     print(data[0].shape, data[1].shape)

    set# load val and train into cache if specified
    if train_loader.dataset.enable_cache:
        print(f'[Train] Filling cache...')
        t0 = time.time()
        for i, data in enumerate(train_loader):
            pass
        num_teeth = train_loader.dataset.num_teeth
        print(f'[Train] Filling cache took {format_duration(time.time()-t0)}',)

    # show stats of num_teeth such as median, mean, min, max, percentile
    print(f"Num teeth stats: median: {np.median(num_teeth)}, mean: {np.mean(num_teeth)}, min: {np.min(num_teeth)}, max: {np.max(num_teeth)}, 25th percentile: {np.percentile(num_teeth, 25)}, 75th percentile: {np.percentile(num_teeth, 75)}")


    # import open3d as o3d
    # for epoch in range(5):
    #     if epoch==0:
    #         train_loader.dataset.set_use_cached_data(True)

    #     t0 = time.time()
    #     for i, data in enumerate(train_loader): 
    #         # visualize data
    #         pcd = o3d.geometry.PointCloud()
    #         pcd.points = o3d.utility.Vector3dVector(data[0][0].cpu().numpy())
    #         o3d.visualization.draw_geometries([pcd])



#     print(f'[Train] {time.time()-t0}',)

        
