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

def compute_normals(mesh):
    normalFaces = np.zeros((mesh.n_faces(), 3))
    points = mesh.points()

    for f in mesh.faces():
        v_it = openmesh.FaceVertexIter(mesh, f)

        v0 = next(v_it).idx()
        v1 = next(v_it).idx()
        v2 = next(v_it).idx()

        normal = np.cross(points[v1,:]-points[v0,:], points[v2,:]-points[v0,:])
        normal = normal/np.linalg.norm(normal)
        normalFaces[f.idx(),:] = normal

    #Computar normales por vertice
    normalVertices = np.zeros((mesh.n_vertices(),3))

    for v in mesh.vertices():
        f_it = openmesh.VertexFaceIter(mesh, v)
        cont = 0
        for f in f_it:
            normalVertices[v.idx(),:] += normalFaces[f.idx(),:]
            cont +=1
        normalVertices[v.idx(),:] = normalVertices[v.idx(),:]/cont
        normalVertices[v.idx(),:] = normalVertices[v.idx(),:]/np.linalg.norm(normalVertices[v.idx(),:])

    return normalVertices

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="", help="")
opt = parser.parse_args()

mesh = openmesh.read_trimesh(opt.file)
points = mesh.points()
normals = compute_normals(mesh)
L = laplacian(mesh)
rho = np.random.normal(size=mesh.n_vertices())*0.02

points = points + np.multiply(np.ones(points.shape), rho[:,np.newaxis])*normals

dt = 0.05
for i in range(500):
    points = points - dt*L.dot(points)

ps.init()
ps_mesh = ps.register_surface_mesh("mesh", points, mesh.face_vertex_indices())
ps.show()