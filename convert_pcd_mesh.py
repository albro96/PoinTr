import open3d as o3d
import pytorch3d as p3d
from pytorch3d.ops import sample_farthest_points as fps
import torch
import glob
import sys
import os.path as op
import os
from easydict import EasyDict
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/storage/share/code/01_scripts/modules/')
from os_tools.import_dir_path import import_dir_path, convert_path

num_points = 1024
create_mesh = False

# pcd_dir_nt = r'O:\data\models\PoinTr\240409_PoinTr-sweep\inference\ckpt-bestfaithful-sweep-122_test-epoch-410'
pcd_dir_nt = r'O:\data\models\PoinTr\240409_PoinTr-sweep\inference\ckpt-bestblooming-sweep-2_test-epoch-0'


pcd_dir = convert_path(pcd_dir_nt)
fps_dir = op.join(pcd_dir, f'fps-{num_points}')
mesh_dir = op.join(pcd_dir, 'mesh')

os.makedirs(mesh_dir, exist_ok=True)
os.makedirs(fps_dir, exist_ok=True)
# load all files in the directory
corr_file_paths = sorted(glob.glob(op.join(pcd_dir, '*corr.pcd')))

data_dir = EasyDict()
for corr_file_path in corr_file_paths:
    num = op.basename(corr_file_path).split('-')[0]
    data_dir[num] = EasyDict({'corr': corr_file_path, 'gt': corr_file_path.replace('corr', 'gt'), 'pred': corr_file_path.replace('corr', 'pred')})


for data_num, pcds in data_dir.items():
    data_type = 'pred'

    assert data_type in pcds.keys()

    pcd_pred = o3d.io.read_point_cloud(pcds[data_type])



    if data_type == 'pred':
        tensor_pred = torch.tensor(pcd_pred.points).unsqueeze(0)
        print(tensor_pred.shape)
        tensor_pred_red = fps(tensor_pred, K=num_points)[0].squeeze(0)
        # print(tensor_pred_red.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tensor_pred_red.numpy())

        o3d.io.write_point_cloud(op.join(fps_dir, f'{data_num}-{data_type}.pcd'), pcd)
    else:
        pcd = pcd_pred

    if create_mesh:
        pcd.normals = o3d.utility.Vector3dVector(np.zeros(
            (1, 3)))  # invalidate existing normals

        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(3)

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            meshSurface, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
        meshSurface.compute_vertex_normals()

        bbox = pcd.get_axis_aligned_bounding_box()
        p_mesh_crop = meshSurface.crop(bbox)

        print('remove low density vertices')
        vertices_to_remove = densities < np.quantile(densities, 1e-10)
        meshSurface.remove_vertices_by_mask(vertices_to_remove)

        arDen = np.asarray(densities)
        # get colormap
        colDen = plt.get_cmap('viridis')((arDen - arDen.min()) / (arDen.max() - arDen.min()))
        colDen = colDen[:, :3]
        # meshDen = meshSurface but with colors for density
        meshDen = o3d.geometry.TriangleMesh()
        meshDen.vertices = meshSurface.vertices
        meshDen.triangles = meshSurface.triangles
        meshDen.triangle_normals = meshSurface.triangle_normals
        meshDen.compute_vertex_normals()
        meshDen.vertex_colors = o3d.utility.Vector3dVector(colDen)

        watertight = meshDen.is_watertight()
        print(f"watertight: {watertight}")

        o3d.io.write_triangle_mesh(op.join(mesh_dir, f'{data_num}-{data_type}.stl'), meshDen)


# load the pcd as tensor

