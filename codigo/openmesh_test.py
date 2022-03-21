import openmesh
import argparse
import polyscope as ps
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="", help="")
opt = parser.parse_args()

mesh = openmesh.read_trimesh(opt.file)

ps.init()
ps_mesh = ps.register_surface_mesh("mesh", mesh.points(), mesh.face_vertex_indices())
ps.show()

