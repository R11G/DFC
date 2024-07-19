import os

import torch
import trimesh as tm
import pytorch_kinematics as pk
import pytorch_volumetric as pv

class RobotModel:
    def __init__(self, path='data/bhand_model/robots/bhand_model.URDF'):
        self.path = path
        self.device = torch.device('cuda')
        device = self.device
        chain = pk.build_chain_from_urdf(open(path).read())
        chain = chain.to(device=device)
        self.n_joints = len(chain.get_joints())
        self.sdf = pv.RobotSDF(chain, path_prefix=os.path.dirname(os.path.abspath(__file__)))
        # add rotation and translation dims (these are applied to meshmodel however)
        self.code_length = self.n_joints+9
        
    def distandgrad(self, x, z):
        joints = z[:,9:]
        self.sdf.set_joint_configuration(joints)
        d,g = self.sdf(x) # B x B x N (x 3). Calcs all batched contacts on all batched joint configs
        # possibly inefficient?
        d = torch.diagonal(d, 0) # N x B
        g = torch.diagonal(g, 0) # N x 3 x B
        d = torch.transpose(d, 0, 1) # B x N
        g = torch.transpose(g, 0, 1)
        g = torch.transpose(g, 0, 2) # B x N x 3
        return -d,g # negative distance because Epen needs vals flipped

    # calculate Eprior
    def prior(self, z):
      # the 9 is ortho6d+translation, so only gets the joint positions
      return torch.norm(z[:,9:], dim=-1)
    

    #unused
    def closest_point(self, obj_code, x):
        distance = self.distance(obj_code, x)
        gradient = self.gradient(x, distance)
        normal = gradient.clone()
        count = 0
        while torch.abs(distance).mean() > 0.003 and count < 100:
            x = x - gradient * distance * 0.5
            distance = self.distance(obj_code, x)
            gradient = self.gradient(x, distance)
            count += 1
        return x.detach(), normal