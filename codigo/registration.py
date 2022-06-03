import open3d as o3d
import numpy as np
import copy
import polyscope as ps


def draw(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)

    ps.init()
    src = np.asarray(source_temp.points)
    tg = np.asarray(target_temp.points)
    ps_cloud1 = ps.register_point_cloud("p1", src)
    ps_cloud2 = ps.register_point_cloud("p2", tg)
    ps.show()


def point_to_point_icp(source, target, threshold, trans_init):
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation, "\n")
    draw(source, target, reg_p2p.transformation)


def point_to_plane_icp(source, target, threshold, trans_init):
    print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation, "\n")
    draw(source, target, reg_p2l.transformation)


if __name__ == "__main__":
    pcd_data = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(pcd_data.paths[0])
    target = o3d.io.read_point_cloud(pcd_data.paths[1])

    #draw(source, target, np.eye(4))
    
    threshold = 0.02
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    #draw(source, target, trans_init)


    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation, "\n")

    #point_to_point_icp(source, target, threshold, trans_init)
    point_to_plane_icp(source, target, threshold, trans_init)