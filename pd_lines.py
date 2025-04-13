import open3d as o3d
import numpy as np
import argparse
import os
import random
import itertools
from scipy.spatial import ConvexHull

def read_point_cloud_auto(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    if path.endswith(".ply") or path.endswith(".pcd"):
        return o3d.io.read_point_cloud(path)
    else:
        raise ValueError("仅支持 .ply 或 .pcd 文件")

def color_point_cloud_by_z_gray(pcd):
    points = np.asarray(pcd.points)
    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_norm = (z_vals - z_min) / (z_max - z_min + 1e-8)
    gray_colors = np.stack([z_norm, z_norm, z_norm], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(gray_colors)
    return pcd

def segment_planes(pcd, max_planes=6, distance_threshold=0.01, min_ratio=0.01):
    planes = []
    equations = []
    rest = pcd
    total_points = len(rest.points)

    for i in range(max_planes):
        if len(rest.points) < total_points * min_ratio:
            break
        model, inliers = rest.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=3,
                                            num_iterations=1000)
        inlier_cloud = rest.select_by_index(inliers)
        rest = rest.select_by_index(inliers, invert=True)
        color = [random.random(), random.random(), random.random()]
        inlier_cloud.paint_uniform_color(color)
        planes.append(inlier_cloud)
        equations.append(model)
        a, b, c, d = model
        print(f"平面 {i+1} 方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    return planes, rest, equations

def group_planes_by_normal(equations, angle_threshold_deg=10):
    from sklearn.cluster import DBSCAN
    normals = np.array([eq[:3] / np.linalg.norm(eq[:3]) for eq in equations])
    eps = np.cos(np.deg2rad(180 - angle_threshold_deg))
    similarity = np.clip(np.dot(normals, normals.T), -1.0, 1.0)
    dist_matrix = 1.0 - np.abs(similarity)
    clustering = DBSCAN(eps=1 - eps, min_samples=1, metric='precomputed')
    labels = clustering.fit_predict(dist_matrix)
    return labels

def estimate_box_from_planes(equations, pcd=None):
    labels = group_planes_by_normal(equations)
    box_planes = []
    for lbl in np.unique(labels):
        group = [equations[i] for i in range(len(equations)) if labels[i] == lbl]
        if len(group) < 2:
            continue
        group_sorted = sorted(group, key=lambda x: x[3])
        box_planes.append((group_sorted[0], group_sorted[-1]))

    normals = [np.array(eq[:3]) / np.linalg.norm(eq[:3]) for eq in equations]

    if len(box_planes) != 3:
        print("⚠️ 正交面组合失败，尝试自动选出方向最分散的三对平面")
        candidates = []
        for i, ni in enumerate(normals):
            for j, nj in enumerate(normals):
                if j <= i: continue
                for k, nk in enumerate(normals):
                    if k <= j: continue
                    score = np.abs(np.dot(ni, nj)) + np.abs(np.dot(ni, nk)) + np.abs(np.dot(nj, nk))
                    candidates.append((score, [i, j, k]))

        if not candidates:
            return None

        best = min(candidates, key=lambda x: x[0])[1]
        selected_normals = [normals[idx] for idx in best]
        print("选择平面组方向夹角（cos值）:")
        for a, b in itertools.combinations(selected_normals, 2):
            print(f"  cos(theta): {np.dot(a, b):.4f}")

        directions = [normals[idx] for idx in best]
    else:
        directions = [np.array(pair[0][:3]) / np.linalg.norm(pair[0][:3]) for pair in box_planes]

    if pcd is None:
        print("❌ 缺少点云数据用于投影生成 box")
        return None

    points = np.asarray(pcd.points)
    center = np.mean(points, axis=0)

    # 构造局部坐标轴
    axes = np.stack(directions)
    proj = (points - center) @ axes.T  # shape (N, 3)
    proj_min = proj.min(axis=0)
    proj_max = proj.max(axis=0)

    corners = []
    for i in [0,1]:
        for j in [0,1]:
            for k in [0,1]:
                local = np.array([
                    proj_min[0] if i == 0 else proj_max[0],
                    proj_min[1] if j == 0 else proj_max[1],
                    proj_min[2] if k == 0 else proj_max[2]
                ])
                pt = center + local @ axes
                corners.append(pt)

    mesh = o3d.geometry.LineSet()
    mesh.points = o3d.utility.Vector3dVector(np.array(corners))
    lines = [
        [0,1], [0,2], [0,4],
        [1,3], [1,5],
        [2,3], [2,6],
        [3,7],
        [4,5], [4,6],
        [5,7],
        [6,7]
    ]
    mesh.lines = o3d.utility.Vector2iVector(lines)
    mesh.paint_uniform_color([1, 0, 1])

    extents = proj_max - proj_min
    print(f"→ 拟合 Box 尺寸: {extents[0]:.4f} x {extents[1]:.4f} x {extents[2]:.4f} mm")

    return mesh

def intersect_three_planes(p1, p2, p3, tol=1e-3):
    """
    三平面近似交点（最小二乘），避免共面或数值不稳定报错
    """
    A = np.array([p1[:3], p2[:3], p3[:3]])
    b = -np.array([p1[3], p2[3], p3[3]])

    if np.linalg.matrix_rank(A) >= 2:
        try:
            pt, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            return pt
        except:
            return None
    return None

def apply_and_combine(source_path, target_path, matrix_file="transformation_matrix.npy"):
    print("[INFO] Loading point clouds...")
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    print(f"[INFO] Loading transformation matrix from '{matrix_file}'...")
    transformation = np.load(matrix_file)
    print("Transformation Matrix:\n", transformation)

    print("[INFO] Applying transformation to source point cloud...")
    source.transform(transformation)

    print("[INFO] Combining transformed source with target...")
    combined = source + target

    combined_file = "combined_cloud_from_matrix.pcd"
    o3d.io.write_point_cloud(combined_file, combined)
    print(f"[INFO] Combined point cloud saved to '{combined_file}'")

    return combined_file

def main(file_path=None, source_file=None, target_file=None,
         max_planes=6, threshold=0.01,
         nb_neighbors=30, std_ratio=1.0):  # NEW: 加入默认滤波参数

    if source_file and target_file:
        file_path = apply_and_combine(source_file, target_file)
    elif not file_path:
        print("提供 --file 或 --pair")
        return
    elif source_file and target_file and file_path:
        print("不要同时提供 --file 和 --pair")
        return
    
    pcd = read_point_cloud_auto(file_path)
    print(f"读取点云: {file_path}，共 {np.asarray(pcd.points).shape[0]} 个点")

    # NEW: 统计滤波，去除孤立点（飞点）
    print("[INFO] 执行统计离群点移除...")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"[INFO] 剩余点数: {len(pcd.points)}")

    pcd = color_point_cloud_by_z_gray(pcd)

    planes, remaining, equations = segment_planes(pcd, max_planes=max_planes, distance_threshold=threshold)
    box_mesh = estimate_box_from_planes(equations, pcd=pcd)

    geometry_list = [pcd]
    geometry_list.extend(planes)
    if box_mesh:
        geometry_list.append(box_mesh)

    o3d.visualization.draw_geometries(geometry_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云包围盒拟合（基于多个提取面）")
    parser.add_argument("--pair", nargs=2, metavar=('SOURCE', 'TARGET'), help="输入一对点云文件用于配准")
    parser.add_argument("--max_planes", type=int, default=6, help="最大提取平面数量")
    parser.add_argument("--threshold", type=float, default=0.05, help="平面提取距离阈值")
    parser.add_argument("--file", type=str, help="输入单个点云文件（.ply 或 .pcd）")
    parser.add_argument("--nb_neighbors", type=int, default=50, help="统计滤波的邻居数")
    parser.add_argument("--std_ratio", type=float, default=0.05, help="统计滤波的标准差比例")
    args = parser.parse_args()

    main(file_path=args.file,
     source_file=args.pair[0] if args.pair else None,
     target_file=args.pair[1] if args.pair else None,
     max_planes=args.max_planes,
     threshold=args.threshold,
     nb_neighbors=args.nb_neighbors,      # NEW
     std_ratio=args.std_ratio)            # NEW
