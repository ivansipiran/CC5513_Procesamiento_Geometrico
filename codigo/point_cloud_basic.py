import numpy as np
import polyscope as ps
ps.init()

# generate some points
points = np.random.rand(1000, 3)
vals = np.random.rand(1000)
vecs = np.random.rand(1000, 3)

# visualize!
ps_cloud = ps.register_point_cloud("my points", points)
#ps_cloud.add_scalar_quantity("vals", vals)
ps_cloud.add_vector_quantity("vecs", vecs)

ps.show()
