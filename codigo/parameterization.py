import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import eigs, spsolve
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

def compute_boundary(F, numVert):
    numFaces = F.shape[0]

    A = np.zeros((numVert, numVert))

    for i in range(numFaces):
        f = F[i,:]
        A[f[0], f[1]] = A[f[0], f[1]] + 1
        A[f[0], f[2]] = A[f[0], f[2]] + 1
        A[f[2], f[1]] = A[f[2], f[1]] + 1
    
    A = A + A.T
    
    flag = 0
    
    for i in range(numVert):
        #Buscar algún elemento con valor 1
        for j in range(numVert):
            if A[i,j] == 1:
                boundary = [j, i]
                flag = 1
            break
        if flag == 1:
            break
    #Segundo elemento de boundary
    s = boundary[1]
    i = 1

    while i < numVert:
        #Buscar algún elemento con valor 1 en la fila s
        vals = []
        for j in range(numVert):
            if A[s,j] == 1:
                vals.append(j)
        
        assert(len(vals)==2)
        if vals[0] == boundary[i-1]:
            s = vals[1]
        else:
            s = vals[0]
        
        if s!=boundary[0]:
            boundary.append(s)
        else:
            break
        i = i + 1
    return boundary

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="", help="")
opt = parser.parse_args()

mesh = openmesh.read_trimesh(opt.file)
points = mesh.points()
numVert = points.shape[0]
boundary = compute_boundary(mesh.face_vertex_indices(), points.shape[0])
numBound = len(boundary)
d = 0
lastBound = boundary[-1]

for i in boundary:
    d = d + np.linalg.norm(points[i,:] - points[lastBound,:])
    lastBound = i

print("Chord length:", d)
vb = points[boundary,:]
sel = list(range(1,numBound))
sel.append(0)
vb2 = vb[sel,:]

D = np.cumsum(np.linalg.norm(vb2-vb, axis=1))
t = (D - D[0])/d
xy_boundary = np.hstack([np.expand_dims(np.cos(2*np.pi*t),axis=1), np.expand_dims(np.sin(2*np.pi*t),axis=1)])
print(xy_boundary)

L = laplacian(mesh)

for i in boundary:
    L[i,:] = 0
    L[i,i] = 1

x = np.zeros((numVert,1))
for i,b in enumerate(boundary):
    x[b,0] = xy_boundary[i,0]
xx = spsolve(L, x)

y = np.zeros((numVert,1))
for i,b in enumerate(boundary):
    y[b,0] = xy_boundary[i,1]
yy = spsolve(L, y)

coord = np.column_stack((xx,yy, np.ones(numVert)))
print(coord.shape)



#L = laplacian(mesh)
#rho = np.random.normal(size=mesh.n_vertices())*0.02

#points = points + np.multiply(np.ones(points.shape), rho[:,np.newaxis])*normals

#dt = 0.05
#for i in range(500):
#    points = points - dt*L.dot(points)

ps.init()
ps_mesh = ps.register_surface_mesh("mesh", coord, mesh.face_vertex_indices())
ps.show()