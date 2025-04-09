import open3d as o3d
import numpy as np
import copy
import sys
import os


def load_and_preprocess(path, voxel_size):
    if not os.path.exists(path):
        raise FileNotFoundError(f"点云文件未找到: {path}")
    
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"点云文件为空或格式不正确: {path}")
    
    print(f"Loaded: {path} 共有 {np.asarray(pcd.points).shape[0]} 个点")

    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2, max_nn=30))

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )

    return pcd, pcd_down, fpfh


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])  # 红色
    target_temp.paint_uniform_color([0, 1, 0])  # 绿色
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def main():
    if len(sys.argv) != 3:
        print("用法: python point_cloud_registration.py source.pcd target.pcd")
        sys.exit(1)

    source_path = sys.argv[1]
    target_path = sys.argv[2]
    voxel_size = 1.0  # 可根据点云密度调整

    # 加载与预处理
    source_raw, source_down, source_fpfh = load_and_preprocess(source_path, voxel_size)
    target_raw, target_down, target_fpfh = load_and_preprocess(target_path, voxel_size)

    print("Downsampled source:", len(source_down.points))
    print("Downsampled target:", len(target_down.points))

    print("Source FPFH shape:", np.asarray(source_fpfh.data).shape)
    print("Target FPFH shape:", np.asarray(target_fpfh.data).shape)

    draw_registration_result(source_down, target_down, np.identity(4))

    # 粗略配准
    print("Running RANSAC...")
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=False,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
    )

    print("RANSAC Transformation:")
    print(result_ransac.transformation)
    draw_registration_result(source_raw, target_raw, result_ransac.transformation)

    source_raw.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_raw.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 精配准（ICP）
    print("Running ICP...")
    result_icp = o3d.pipelines.registration.registration_icp(
        source_raw, target_raw,
        max_correspondence_distance=voxel_size * 0.5,
        init=result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    print("ICP Transformation:")
    print(result_icp.transformation)
    draw_registration_result(source_raw, target_raw, result_icp.transformation)


if __name__ == "__main__":
    main()