# gicp_registration.py
import open3d as o3d
import numpy as np
import copy
import sys
import os

def load_initial_transform(txt_path):
    print(f"[INFO] Loading initial transform from: {txt_path}")
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        matrix = []
        for line in lines:
            row = [float(x) for x in line.strip().split()]
            matrix.append(row)
        return np.array(matrix)

def preprocess_point_cloud(pcd, voxel_size):
    print(f"[INFO] Downsampling with voxel size: {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print("[INFO] Estimating normals")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2.0, max_nn=30
        )
    )
    return pcd_down

def run_gicp(source_path, target_path, init_transform_path=None, voxel_size=0.1, max_iter=10000):
    print("[INFO] Loading point clouds...")
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)

    print("[INFO] Preparing transformation...")
    if init_transform_path and os.path.exists(init_transform_path):
        trans_init = load_initial_transform(init_transform_path)
    else:
        print("[WARN] No valid initial transform provided. Using identity matrix.")
        trans_init = np.identity(4)

    # 可视化初始对齐效果
    print("[INFO] Visualizing initial alignment...")
    source_down_for_vis = copy.deepcopy(source_down)
    source_down_for_vis.transform(trans_init.copy())

    o3d.visualization.draw_geometries([
        source_down_for_vis.paint_uniform_color([1, 0, 0]),
        target_down.paint_uniform_color([0, 1, 0])
    ], window_name="Initial Alignment Preview")

    print("[INFO] Running GICP...")
    threshold = voxel_size * 15.5
    result = o3d.pipelines.registration.registration_generalized_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )

    print("[INFO] GICP finished.")
    print("Transformation Matrix:\n", result.transformation)

    # 保存转换矩阵
    matrix_file = "transformation_matrix.npy"
    np.save(matrix_file, result.transformation)
    print(f"[INFO] Transformation matrix saved to '{matrix_file}'")

    # 应用变换
    print("[INFO] Applying transformation to source...")
    source.transform(result.transformation)

    o3d.visualization.draw_geometries([
        source.paint_uniform_color([1, 0, 0]),  # 红色源点云
        target.paint_uniform_color([0, 1, 0])   # 绿色目标点云
    ], window_name="Aligned Preview")

    # 合并点云
    print("[INFO] Combining source and target point clouds...")
    combined = source + target

    # 保存合并后的点云
    combined_file = "combined_registered_cloud.pcd"
    o3d.io.write_point_cloud(combined_file, combined)
    print(f"[INFO] Combined point cloud saved to '{combined_file}'")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gicp_registration.py source.pcd target.pcd [init_matrix.txt]")
        sys.exit(1)

    source_file = sys.argv[1]
    target_file = sys.argv[2]
    init_matrix_file = sys.argv[3] if len(sys.argv) >= 4 else None

    run_gicp(source_file, target_file, init_matrix_file)