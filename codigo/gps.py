import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import eigs
import openmesh
import argparse
import os
import polyscope as ps
import trimesh
from sklearn.cluster import KMeans

import signature
import laplace
import matplotlib.pyplot as plt

def compute_gps(filename, args):
    name = os.path.splitext(filename)[0]
    if os.path.exists(name + '.npz'):
        extractor = signature.SignatureExtractor(path=name+'.npz')
    else:
        mesh = trimesh.load(filename)
        extractor = signature.SignatureExtractor(mesh, 100, args.approx)
        np.savez_compressed(name+'.npz', evals=extractor.evals, evecs=extractor.evecs)
    
    gps = extractor.evecs[:,1:25]/np.sqrt(np.tile(extractor.evals[1:25], (extractor.evecs.shape[0],1)))
    return gps
    

parser = argparse.ArgumentParser(description='Mesh signature visualization')
parser.add_argument('--n_basis', default='100', type=int, help='Number of basis used')
parser.add_argument('--f_size', default='128', type=int, help='Feature size used')
parser.add_argument('--approx', default='cotangens', choices=laplace.approx_methods(), type=str, help='Laplace approximation to use')
parser.add_argument('--laplace', help='File holding laplace spectrum')
parser.add_argument('--kernel', type=str, default='heat', help='Feature type to extract. Must be in [heat, wave]')

args = parser.parse_args()

file = 'cat0.off'

vids = [18047, 14680, 25201, 21688]

gps = compute_gps(file, args)
kmeans = KMeans(n_clusters=7, random_state=0).fit(gps)

mesh = openmesh.read_trimesh(file)
ps.init()
ps_mesh = ps.register_surface_mesh("mesh", mesh.points(), mesh.face_vertex_indices())
ps_mesh.add_scalar_quantity("distances", kmeans.labels_)
ps.show()