import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import eigs
import openmesh
import argparse
import os
import polyscope as ps
import trimesh

import signature
import laplace
import matplotlib.pyplot as plt

def compute_eigen(filename, args):
    name = os.path.splitext(filename)[0]
    if os.path.exists(name + '.npz'):
        extractor = signature.SignatureExtractor(path=name+'.npz')
    else:
        mesh = trimesh.load(filename)
        extractor = signature.SignatureExtractor(mesh, 300, args.approx)
        np.savez_compressed(name+'.npz', evals=extractor.evals, evecs=extractor.evecs)
    
    return extractor.evals, extractor.evecs

parser = argparse.ArgumentParser(description='Mesh signature visualization')
parser.add_argument('--n_basis', default='100', type=int, help='Number of basis used')
parser.add_argument('--f_size', default='128', type=int, help='Feature size used')
parser.add_argument('--approx', default='cotangens', choices=laplace.approx_methods(), type=str, help='Laplace approximation to use')
parser.add_argument('--laplace', help='File holding laplace spectrum')
parser.add_argument('--kernel', type=str, default='heat', help='Feature type to extract. Must be in [heat, wave]')

args = parser.parse_args()

file1 = 'cat0.off'
vid = 18047

evals, evecs = compute_eigen(file1, args)
distances = ((np.tile(evecs[vid,:], (evecs.shape[0],1)) - evecs[:,:])**2)
dist = []

for i in range(evecs.shape[0]):
    sum = 0.0
    for j in range(1,evecs.shape[1]):
        sum += distances[i,j]/(evals[j]**2)
    dist.append(sum)


mesh = openmesh.read_trimesh(file1)
ps.init()
ps_mesh = ps.register_surface_mesh("mesh", mesh.points(), mesh.face_vertex_indices())
ps_mesh.add_distance_quantity("distances", np.asarray(dist))
#ps_mesh.add_scalar_quantity("distances", distances)
ps.show()