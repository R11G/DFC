import argparse
import random

import numpy as np
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--n_contact', default=5, type=int)
parser.add_argument('--max_physics', default=10000, type=int)
parser.add_argument('--max_refine', default=1000, type=int)
parser.add_argument('--hand_model', default='mano', type=str)
parser.add_argument('--obj_model', default='sphere', type=str)
parser.add_argument('--langevin_probability', default=0.85, type=float)
parser.add_argument('--hprior_weight', default=1, type=float)
parser.add_argument('--noise_size', default=0.1, type=float)
parser.add_argument('--output_dir', default='synthesis', type=str)
args = parser.parse_args()
d = 'cuda' if torch.cuda.is_available() else 'cpu'
# set random seeds
np.seterr(all='raise')
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from utils.MeshModel import MeshModel
from utils.Losses import FCLoss
from utils.RobotModel import RobotModel
from utils.PenetrationModel import PenetrationModel
from utils.PhysicsGuide import PhysicsGuide

# prepare models
if args.obj_model == 'sphere':
    mesh_model = MeshModel(n_points=100)
else:
    raise NotImplementedError()

robot_model = RobotModel()

fc_loss_model = FCLoss(robot_model=robot_model)
penetration_model = PenetrationModel(mesh_model=mesh_model, robot_model=robot_model)

physics_guide = PhysicsGuide(mesh_model, robot_model, penetration_model, fc_loss_model, args)

accept_history = []

z = torch.normal(0, 1, [args.batch_size, robot_model.code_length], device=d, dtype=torch.float32, requires_grad=True)
contact_point_indices = torch.randint(0, mesh_model.n_pts, [args.batch_size, args.n_contact], device=d, dtype=torch.long)

# optimize hand pose and contact map using physics guidance
energy, grad, verbose_energy = physics_guide.initialize(z, contact_point_indices)
linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = verbose_energy
accept = ((force_closure < 0.5) * (penetration < 0.02) * (surface_distance < 0.02)).float()
for physics_step in range(args.max_physics):
    energy, grad, z, contact_point_indices, verbose_energy = physics_guide.optimize(energy, grad, z, contact_point_indices, verbose_energy)
    linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = verbose_energy
    val = (force_closure + penetration + surface_distance).float()
    accept = ((force_closure < 0.5) * (penetration < 0.02) * (surface_distance < 0.02)).float()
    _accept = accept.sum().detach().cpu().numpy()
    accept_history.append(_accept)
    if physics_step % 100 == 0:
        print('optimize', physics_step, _accept, val)

for refinement_step in range(args.max_refine):
    energy, grad, z, contact_point_indices, verbose_energy = physics_guide.refine(energy, grad, z, contact_point_indices, verbose_energy)
    linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = verbose_energy
    val = (force_closure + penetration + surface_distance).float()
    accept = ((force_closure < 0.5) * (penetration < 0.02) * (surface_distance < 0.02)).float()
    _accept = accept.sum().detach().cpu().numpy()
    accept_history.append(_accept)
    if refinement_step % 100 == 0:
        print('refine', refinement_step, _accept, val)


#os.makedirs('%s/%s-%s-%d-%d'%(args.output_dir, args.hand_model, args.obj_model, args.n_contact, args.batch_size), exist_ok=True)
# TODO: find new visualization tool to replace below
"""
for a in torch.where(accept)[0]:
    a = a.detach().cpu().numpy()
    hand_verts = physics_guide.hand_model.get_vertices(z)[a].detach().cpu().numpy()
    hand_faces = physics_guide.hand_model.faces
    if args.obj_model == "sphere":
        sphere = tm.primitives.Sphere(radius=object_code[a].detach().cpu().numpy())
        fig = go.Figure([utils.visualize_plotly.plot_hand(hand_verts, hand_faces), utils.visualize_plotly.plot_obj(sphere)])
    else:
        mesh = object_model.get_obj_mesh(object_idx[[a]].detach().cpu().numpy())
        fig = go.Figure([utils.visualize_plotly.plot_hand(hand_verts, hand_faces), utils.visualize_plotly.plot_obj(mesh)])
    fig.write_html('%s/%s-%s-%d-%d/fig-%d.html'%(args.output_dir, args.hand_model, args.obj_model, args.n_contact, args.batch_size, a))"""

