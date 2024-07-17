import polyscope as ps
import gpytoolbox as gpy
import numpy as np
import scipy as sp

# number of points in x, y
nx, ny = 5, 5

# create a grid
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

Z = np.zeros_like(X)
V = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

def index_V(i, j):
    return i + j * nx

# loop over all cells and create quad meshes
F = []
for i in range(nx - 1):
    for j in range(ny - 1):
        # add the faces in counter clockwise order
        F.append([index_V(i, j), index_V(i + 1, j), index_V(i + 1, j + 1)])
        F.append([index_V(i, j), index_V(i + 1, j + 1), index_V(i, j + 1)])

F = np.array(F)
# N = gpy.per_face_normals(V,F)

# warp the vertices
from warp_function_1 import *
V_warped = warp_function(V[:, :2])
assert V_warped.shape == V.shape

from chamfer_distance import *

# # compute the chamfer distance with all V
# cd, cpv, cpf = chamfer_distance(V, V_warped)

# sample points on original grid
n_samples = 10
V_sampled = V[np.random.choice(V.shape[0], n_samples)]
cd, cpv, cpe = chamfer_distance(V_sampled, V_warped)

# make all the points green initially
cp_colours = np.tile([0, 1, 0], cpv.shape[0]).reshape(-1, 3)
assert cp_colours.shape[0] == cpv.shape[0] and cp_colours.shape[1] == 3
# colour it red if it was sampled
for i in cpe:
    cp_colours[i[0], :] = np.array([1, 0, 0])
    cp_colours[i[1], :] = np.array([1, 0, 0])

ps.init()
ps.register_point_cloud("points", V)
ps_mesh = ps.register_surface_mesh("grid", V, F)
ps_warped = ps.register_surface_mesh("warped grid", V_warped, F)

ps_net = ps.register_curve_network("closest points", cpv, cpe)
ps_net.add_color_quantity("cp_colours", cp_colours, enabled=True)

ps.show()