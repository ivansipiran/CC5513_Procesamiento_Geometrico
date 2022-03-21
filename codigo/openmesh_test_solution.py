import openmesh
import argparse
import polyscope as ps
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="", help="")
opt = parser.parse_args()

mesh = openmesh.read_trimesh(opt.file)
#print(mesh.n_vertices())
#print(mesh.n_faces())

#Iterar sobre vertices
#for v in mesh.vertices():
#    print(v.idx())

#Iterar sobre faces
#for f in mesh.faces():
#    print(f)

vals = np.zeros(mesh.n_vertices())
valence = np.zeros(mesh.n_vertices())

# Computar vertices frontera
#for v in mesh.vertices():
#    if mesh.is_boundary(v):
#        vals[v.idx()] = 1

#Computar la valencia
#for v in mesh.vertices():
#    v_it = openmesh.VertexVertexIter(mesh, v)
#    cont = 0
#    for x in v_it:
#        cont += 1
#    valence[v.idx()] = cont

#for f in mesh.faces():
#    v_it = openmesh.FaceVertexIter(mesh, f)
#    L = []
#    for v in v_it:
#        L.append(v.idx())
#    print(L)

# Computar normales por cara
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

ps.init()
ps_mesh = ps.register_surface_mesh("mesh", mesh.points(), mesh.face_vertex_indices())
#ps_mesh.add_scalar_quantity("vals", valence)
#ps_mesh.add_vector_quantity("normalFaces", normalFaces, defined_on="faces")
ps_mesh.add_vector_quantity("normalVertices", normalVertices)
ps.show()

