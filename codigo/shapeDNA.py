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

def compute_spectrum(filename, args):
    name = os.path.splitext(filename)[0]
    if os.path.exists(name + '.npz'):
        extractor = signature.SignatureExtractor(path=name+'.npz')
    else:
        mesh = trimesh.load(filename)
        extractor = signature.SignatureExtractor(mesh, 100, args.approx)
        np.savez_compressed(name+'.npz', evals=extractor.evals, evecs=extractor.evecs)
    
    return extractor.evals[0:10]

parser = argparse.ArgumentParser(description='Mesh signature visualization')
parser.add_argument('--n_basis', default='100', type=int, help='Number of basis used')
parser.add_argument('--f_size', default='128', type=int, help='Feature size used')
parser.add_argument('--approx', default='cotangens', choices=laplace.approx_methods(), type=str, help='Laplace approximation to use')
parser.add_argument('--laplace', help='File holding laplace spectrum')
parser.add_argument('--kernel', type=str, default='heat', help='Feature type to extract. Must be in [heat, wave]')

args = parser.parse_args()

filesCat = ['cat0.off','cat1.off','cat2.off','cat3.off','cat4.off','cat5.off','cat6.off','cat7.off','cat8.off','cat9.off','cat10.off', 'cat_sampling.off']
filesHorse=['horse0.off','horse5.off','horse6.off','horse7.off','horse10.off','horse14.off','horse15.off','horse17.off']

shapeDNA = dict()

for f in filesCat:
    shapeDNA[f] = compute_spectrum(f, args)

for f in filesHorse:
    shapeDNA[f] = compute_spectrum(f, args)

for key, value in shapeDNA.items():
    if 'cat' in key:
        plt.plot(value, label=key, color='red')
    else:
        plt.plot(value, label=key, color='blue')
plt.legend()

plt.show()
