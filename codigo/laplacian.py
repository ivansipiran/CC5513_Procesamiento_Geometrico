import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import eigs
import openmesh
import argparse
import os
import polyscope as ps

def myangle(u,v):
    du = np.linalg.norm(u)
    dv = np.linalg.norm(v)

    du = max(du, 1e-8)
    dv = max(dv, 1e-8)

    return np.arccos(np.dot(u,v)/(du*dv))

#Funcion recibe un objeto mesh de OpenMesh
def laplacian(mesh):
    n = mesh.n_vertices()
    print(f"Num. vertices {n}")
    W = lil_matrix((n,n), dtype=np.float)
    print(W.shape)

    points = mesh.points()

    #Para cada vertice
    for i,v in enumerate(mesh.vertices()):
        f_it = openmesh.VertexFaceIter(mesh, v)
        for f in f_it:
            v_it = openmesh.FaceVertexIter(mesh,f)
            L = []
            for vv in v_it:
                if vv.idx()!=i:
                    L.append(vv.idx())
            j = L[0]
            k = L[1]

            vi = points[i,:]
            vj = points[j,:]
            vk = points[k,:]

            alpha = myangle(vk-vi, vk-vj)
            beta = myangle(vj-vi,vj-vk)

            W[i,j] = W[i,j] + 1.0/np.tan(alpha)
            W[i,k] = W[i,k] + 1.0/np.tan(beta)
    
    S = 1.0/W.sum(axis=1)
    print(S.shape)
    W = eye(n,n)-spdiags(np.squeeze(S),0,n,n)*W
    return W

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="", help="")
opt = parser.parse_args()

mesh = openmesh.read_trimesh(opt.file)
ps.init()

print(opt.file[:-3])
if not os.path.exists(opt.file[:-3]+'evals'):
    L0 = laplacian(mesh)
    vals, vecs = eigs(L0, k=60, which='SM')
    np.savetxt(opt.file[:-3]+'evals',np.real(vals))
    np.savetxt(opt.file[:-3]+'evecs',np.real(vecs))
else:
    vals = np.loadtxt(opt.file[:-3]+'evals')
    vecs = np.loadtxt(opt.file[:-3]+'evecs')

ps_mesh = ps.register_surface_mesh("mymesh", mesh.points(), mesh.face_vertex_indices())

for i in range(0,60,6):
    ps_mesh.add_scalar_quantity("scalar"+str(i), vecs[:,i])
#evecs = vecs[:,0:60]
#vecT = evecs.T
#new_points = evecs.dot(vecT.dot(mesh.points()))
#ps_mesh = ps.register_surface_mesh("mymesh", mesh.points(), mesh.face_vertex_indices())
ps.show()    


