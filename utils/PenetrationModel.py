import torch.nn as nn

class PenetrationModel:
  def __init__(self, mesh_model, robot_model):
    self.mesh_model = mesh_model
    self.robot_model = robot_model
    self.relu = nn.ReLU()

  def get_penetration(self, z):
    h2o_distances, g = self.robot_model.distandgrad(self.mesh_model.get_vertices(z), z) # B x V x 1
    penetration = self.relu(h2o_distances)#.squeeze(-1) ours is already B x V
    return penetration
