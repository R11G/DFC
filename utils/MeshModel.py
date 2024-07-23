import os

import numpy as np
import torch
import open3d as o3d
import pytorch_kinematics as pk
import pytorch_volumetric as pv

class MeshModel:
  def __init__(self, path='utils/sphere.stl', n_points=100):
#n_handcode=6, root_rot_mode='ortho6d', robust_rot=False, flat_hand_mean=False,mano_path='data/mano', n_contact=3, scale=120
    #self.scale = scale
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.device = torch.device(d)
    device = self.device
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())
    mesh.scale(1/np.max(mesh.get_max_bound()-mesh.get_min_bound()), mesh.get_center())
    self.sdf = pv.MeshObjectFactory(mesh=mesh)
    self.sdf = pv.MeshSDF(self.sdf)
    self.n_pts = n_points
    self.pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    self.pts = torch.from_numpy(np.asarray(self.pcd.points)).float().to(device)
    self.norms = torch.from_numpy(np.asarray(self.pcd.normals)).float().to(device)
  
  # get point cloud of object
  def get_vertices(self, hand_code):
    B = hand_code.shape[0]
    x = hand_code[:,[3,4,5]] # B x 3
    y = hand_code[:,[6,7,8]] # B x 3
    z = torch.cross(x, y, dim=1) # B x 3
    rot_mat = torch.stack((x,y,z), dim=1) # B x 3 x 3
    pts = self.pts.repeat(B, 1, 1) # B x N x 3
    hand_trans = hand_code[:,:3] # B x 3
    hand_trans = hand_trans.unsqueeze(1) # B x 1 x 3
    hand_trans = hand_trans.repeat(1, self.n_pts, 1) # B x N x 3
    torch.matmul(pts, rot_mat)
    pts = pts + hand_trans
    return pts
  
  # get normals
  def get_surface_normals(self, hand_code):
    B = hand_code.shape[0]
    x = hand_code[:,[3,4,5]] # B x 3
    y = hand_code[:,[6,7,8]] # B x 3
    z = torch.cross(x, y, dim=1) # B x 3
    rot_mat = torch.stack((x,y,z), dim=1) # B x 3 x 3
    norms = self.norms.repeat(B, 1, 1) # B x N x 3
    hand_trans = hand_code[:,:3] # B x 3
    hand_trans = hand_trans.unsqueeze(1) # B x 1 x 3
    hand_trans = hand_trans.repeat(1, self.n_pts, 1) # B x N x 3
    torch.matmul(norms, rot_mat)
    norms = norms + hand_trans
    return norms