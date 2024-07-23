import argparse
import random

import numpy as np
import torch
from datetime import datetime
import time
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--n_contact', default=5, type=int)
parser.add_argument('--max_physics', default=10, type=int)
parser.add_argument('--max_refine', default=10, type=int)
parser.add_argument('--hand_model', default='mano', type=str)
parser.add_argument('--obj_model', default='sphere', type=str)
parser.add_argument('--langevin_probability', default=0.85, type=float)
parser.add_argument('--hprior_weight', default=1, type=float)
parser.add_argument('--noise_size', default=0.1, type=float)
parser.add_argument('--output_dir', default='synthesis', type=str)
parser.add_argument('--n_pcd', default=1000, type=int)
args = parser.parse_args()
d = 'cuda' if torch.cuda.is_available() else 'cpu'
# set random seeds. set to current time
np.seterr(all='raise')
seed = int(round(datetime.now().timestamp()))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

from utils.MeshModel import MeshModel
from utils.Losses import FCLoss
from utils.RobotModel import RobotModel
from utils.PenetrationModel import PenetrationModel
from utils.PhysicsGuide import PhysicsGuide

# prepare models
if args.obj_model == 'sphere':
    mesh_model = MeshModel(n_points=args.n_pcd)
else:
    raise NotImplementedError()

robot_model = RobotModel()

fc_loss_model = FCLoss(robot_model=robot_model)
penetration_model = PenetrationModel(mesh_model=mesh_model, robot_model=robot_model)

physics_guide = PhysicsGuide(mesh_model, robot_model, penetration_model, fc_loss_model, args)

accept_history = []
fchist = []
penhist = []
disthist = []
recordstep = 10
#start = time.time()
# config: B x (J+9) matrix with xyz translation vector, 2x xyz rotation vector, joint rotations
config = torch.normal(0, 1, [args.batch_size, robot_model.code_length], device=d, dtype=torch.float32, requires_grad=True)
contact_point_indices = torch.randint(0, mesh_model.n_pts, [args.batch_size, args.n_contact], device=d, dtype=torch.long)

# optimize hand pose and contact map using physics guidance
energy, grad, verbose_energy = physics_guide.initialize(config, contact_point_indices)
linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = verbose_energy
accept = ((force_closure < 0.5) * (penetration < 0.02) * (surface_distance < 0.02)).float()
for physics_step in range(args.max_physics):
    """end = time.time()
    print(physics_step, end - start)
    start = time.time()"""
    energy, grad, config, contact_point_indices, verbose_energy = physics_guide.optimize(energy, grad, config, contact_point_indices, verbose_energy)
    linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = verbose_energy
    #accept = ((force_closure < 0.5) * (penetration < 0.02) * (surface_distance < 0.02)).float()
    #_accept = accept.sum().detach().cpu().numpy()
    #accept_history.append(_accept)
    if physics_step % recordstep == 0:
        disthist.append(surface_distance.detach().clone().numpy())
        penhist.append(penetration.detach().clone().numpy())
        fchist.append(force_closure.detach().clone().numpy())
    if physics_step % 100 == 0:
        #print('optimize', physics_step, _accept)
        print('fc', force_closure.detach())
        print('pen', penetration)
        print('dist', surface_distance)

for refinement_step in range(args.max_refine):
    #print(refinement_step)
    energy, grad, config, contact_point_indices, verbose_energy = physics_guide.refine(energy, grad, config, contact_point_indices, verbose_energy)
    linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = verbose_energy
    """accept = ((force_closure < 0.5) * (penetration < 0.02) * (surface_distance < 0.02)).float()
    _accept = accept.sum().detach().cpu().numpy()
    accept_history.append(_accept)"""
    if physics_step % recordstep == 0:
        disthist.append(surface_distance.detach().clone().numpy())
        penhist.append(penetration.detach().clone().numpy())
        fchist.append(force_closure.detach().clone().numpy())
    if refinement_step % 100 == 0:
        #print('refine', refinement_step, _accept)
        print('fc', force_closure.detach())
        print('pen', penetration)
        print('dist', surface_distance)

np.savetxt('dist.csv', np.array(disthist), delimiter=',')
np.savetxt('fc.csv', np.array(fchist), delimiter=',')
np.savetxt('pen.csv', np.array(penhist), delimiter=',')
#np.savetxt('accept.csv', np.array(accept_history), delimiter=',')
print('fc', force_closure.detach())
print('pen', penetration)
print('dist', surface_distance)