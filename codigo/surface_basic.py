import open3d as o3d
import polyscope as ps
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='', help='')
opt = parser.parse_args()

mesh = o3d.io.read_triangle_mesh(opt.file)
vertices = np.asarray(mesh.vertices)

triangles = np.asarray(mesh.triangles)

ps.init()
ps_mesh = ps.register_surface_mesh("mesh", vertices, triangles)
ps.show()


