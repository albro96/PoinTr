import pytorch3d as p3d
import pytorch3d.ops as ops
import torch
import glob
import sys
import os.path as op
import os
import numpy as np
from easydict import EasyDict
import open3d as o3d

sys.path.append('/storage/share/code/01_scripts/modules/')
from os_tools.import_dir_path import import_dir_path, convert_path
from pcd_tools.visualizer import ObjectVisualizer

device = torch.device('cuda:0')

pada = import_dir_path()

pcd_dir_nt = r'O:\data\3d-datasets\3DTeethSeg22\data-single\npy_8192\00OMSZGW'
pcd_dir = convert_path(pcd_dir_nt)

save_dir = convert_path(r'O:\nobackup\temp\torch-3d-test')

pcd_full = None

for idx, pcd_path in enumerate(glob.glob(op.join(pcd_dir, '3*.npy'))):
    pcd = torch.tensor(np.load(pcd_path))
    if pcd_full is None:
        pcd_full = pcd
    else:
        pcd_full = torch.cat([pcd_full, pcd], dim=0)


pcd_full_orig = pcd_full.clone()
o3d_pcd = o3d.geometry.PointCloud()
o3d_pcd.points = o3d.utility.Vector3dVector(pcd_full.numpy())
o3d.io.write_point_cloud(op.join(save_dir, 'pcd_full_orig.pcd'), o3d_pcd)

# rotate and translate pcd_full randomly and save the T and R matrix
# create rotation matrix to rotate pcd 44 deg around the y axis
theta = 44
theta_rad = np.deg2rad(theta)
R = torch.tensor([[np.cos(theta_rad), 0, np.sin(theta_rad)], [0, 1, 0], [-np.sin(theta_rad), 0, np.cos(theta_rad)]])
# create translation vector to translate pcd 0.5 units along the x axis
T = torch.tensor([50.0, 0, 0])

# pcd_full = pcd_full[:pcd_full.shape[0]//2,:]

# apply the rotation and translation
pcd_full = torch.matmul(pcd_full, R) + T

o3d_pcd = o3d.geometry.PointCloud()
o3d_pcd.points = o3d.utility.Vector3dVector(pcd_full.numpy())
o3d.io.write_point_cloud(op.join(save_dir, 'pcd_full_rot.pcd'), o3d_pcd)

print(pcd_full.shape)
# Convert tensors to Float
pcd_full = pcd_full.float().to(device)
pcd_full_orig = pcd_full_orig.float().to(device)


R_calc, T_calc, scale_calc = ops.corresponding_points_alignment(pcd_full_orig.unsqueeze(0), pcd_full.unsqueeze(0), estimate_scale=False)

pcd_rot = torch.matmul(pcd_full_orig, R_calc.squeeze(0)) + T_calc.squeeze(0)
pcd_rot = pcd_rot.detach().cpu()
print(R, T)

print(R_calc, T_calc, scale_calc)

# ICPSolution  = ops.iterative_closest_point(pcd_full_orig.unsqueeze(0), pcd_full.unsqueeze(0), max_iterations=100, relative_rmse_thr=1e-6)
# print(ICPSolution.RTs.R, ICPSolution.RTs.T, ICPSolution.rmse)
# pcd_rot = ICPSolution.Xt.squeeze(0).detach().cpu()

o3d_pcd = o3d.geometry.PointCloud()
o3d_pcd.points = o3d.utility.Vector3dVector(pcd_rot.numpy())
o3d.io.write_point_cloud(op.join(save_dir, 'pcd_full_rot_iter.pcd'), o3d_pcd)


