import open3d as o3d
import numpy as np
import time

# down sample voxel_size
voxel_size = 0.002
max_correspondence_distance_coarse = voxel_size * 20
max_correspondence_distance_fine = voxel_size * 2
normal_radius = voxel_size * 10 # voxel_size * 2

#读取ply 点云文件
#source = o3d.io.read_point_cloud("temp/icptest/icp_test1.ply")  #source 为需要配准的点云
#target = o3d.io.read_point_cloud("temp/icptest/icp_test2.ply")  #target 为目标点云

source = o3d.io.read_point_cloud("temp/icptest/runing_path/4.ply")  #source 为需要配准的点云
target = o3d.io.read_point_cloud("temp/icptest/runing_path/3.ply")  #target 为目标点云

source.translate(np.zeros(3), relative=False)
target.translate(np.zeros(3), relative=False)
#print(source.get_center())
#print(target.get_center())

def execute_fast_global_registration(source, target):
    radius_feature = voxel_size * 12.5 # voxel_size * 5
    print("使用搜索半径为{}计算FPFH特征".format(radius_feature))
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    distance_threshold = 5.0 # voxel_size * 5 # voxel_size * 2 # voxel_size * 0.5
    print("基于距离阈值为 %.3f的快速全局配准" % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(source, target,
        source_fpfh, target_fpfh, o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))
    return result

def pairwise_registration(source, target):
    print("apply point - point or plane ICP")
    # 粗配准
    icp_coarse = o3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance_coarse,
                                                             np.identity(4),
                                                             o3d.pipelines.registration.TransformationEstimationPointToPlane())  # TransformationEstimationPointToPlane
    print(icp_coarse.transformation)
    # 精配准
    icp_fine = o3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance_fine,
                                                           icp_coarse.transformation,
                                                           o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(icp_fine.transformation)
    transformation_icp = icp_fine.transformation
    # 从变换矩阵计算信息矩阵
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine, icp_fine.transformation)
    # return transformation_icp, information_icp
    return icp_fine


# colored pointcloud registration
# This is implementation of following paper
# J. Park, Q.-Y. Zhou, V. Koltun,
# Colored Point Cloud Registration Revisited, ICCV 2017
def colored_icp(source, target):
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = voxel_down_sample(source, voxel_size=radius)
        target_down = voxel_down_sample(source, voxel_size=radius)

        print("3-2. Estimate normal.")
        estimate_normals(source_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        estimate_normals(target_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
                
        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
                source_down, target_down, radius, current_transformation,
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                        relative_rmse=1e-6,
                                                        max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)

    return result_icp

#scale点云
source.scale(0.001, center=np.zeros(3))
target.scale(0.001, center=np.zeros(3))
#source.paint_uniform_color([1, 0.706, 0])    #source 为黄色
#target.paint_uniform_color([0, 0.651, 0.929])#target 为蓝色

#为两个点云分别进行outlier removal, down sample and estimate normals
# print("outlier removal")
# processed_source, outlier_index = o3d.geometry.radius_outlier_removal(source, nb_points=16, radius=0.5)
# processed_target, outlier_index = o3d.geometry.radius_outlier_removal(target, nb_points=16, radius=0.5)
processed_source = source
processed_target = target
print("dwonsample")
processed_source = processed_source.voxel_down_sample(voxel_size=voxel_size)
processed_target = processed_target.voxel_down_sample(voxel_size=voxel_size)
print("estimate normals")
processed_source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
processed_target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

#运行icp
print("run registration_icp")
start = time.time()
# reg_p2p = o3d.pipelines.registration.registration_icp(processed_source, processed_target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
# reg_p2p = pairwise_registration(processed_source, processed_target)
# reg_p2p = colored_icp(processed_source, processed_target)
reg_p2p = execute_fast_global_registration(processed_source, processed_target)
print("registration_icp using %.3f s.\n" % (time.time() - start))
print("done registration_icp")

#将我们的矩阵依照输出的变换矩阵进行变换
print(reg_p2p)
print(reg_p2p.transformation)
processed_source.transform(reg_p2p.transformation)

reg_p2p = pairwise_registration(processed_source, processed_target)
print("done icp refining")
processed_source.transform(reg_p2p.transformation)

#创建一个 o3d.visualizer class
vis = o3d.visualization.Visualizer()
vis.create_window()

#将两个点云放入visualizer
vis.add_geometry(processed_source)
vis.add_geometry(processed_target)

#让visualizer渲染点云
vis.update_renderer()

vis.run()
