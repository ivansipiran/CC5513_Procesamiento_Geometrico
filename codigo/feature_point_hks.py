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

def compute_hks(filename, args):
    name = os.path.splitext(filename)[0]
    if os.path.exists(name + '.npz'):
        extractor = signature.SignatureExtractor(path=name+'.npz')
    else:
        mesh = trimesh.load(filename)
        extractor = signature.SignatureExtractor(mesh, 100, args.approx)
        np.savez_compressed(name+'.npz', evals=extractor.evals, evecs=extractor.evecs)
    
    return extractor.heat_signatures(1, times=[0.1])

parser = argparse.ArgumentParser(description='Mesh signature visualization')
parser.add_argument('--n_basis', default='100', type=int, help='Number of basis used')
parser.add_argument('--f_size', default='128', type=int, help='Feature size used')
parser.add_argument('--approx', default='cotangens', choices=laplace.approx_methods(), type=str, help='Laplace approximation to use')
parser.add_argument('--laplace', help='File holding laplace spectrum')
parser.add_argument('--kernel', type=str, default='heat', help='Feature type to extract. Must be in [heat, wave]')

args = parser.parse_args()

file1 = 'cat0.off'

hks1 = compute_hks(file1, args)
#Hacer cómputo de local maximo
mesh = openmesh.read_trimesh(file1)
points = mesh.points()
featurePoints = []

for v in mesh.vertices():
    v_it = openmesh.VertexVertexIter(mesh, v)
    flag = True
    for w in v_it:
        if hks1[w.idx()] > hks1[v.idx()]:
            flag = False
            break
    if flag:
        featurePoints.append(points[v.idx(),:])

fPoints = np.vstack(featurePoints)
print(fPoints.shape)


ps.init()
ps_mesh = ps.register_surface_mesh("mesh", mesh.points(), mesh.face_vertex_indices())
ps_mesh.add_scalar_quantity("distances", hks1.squeeze())

pc = ps.register_point_cloud("points", fPoints)
ps.show()