# apply_transformation_and_combine.py
import open3d as o3d
import numpy as np
import sys

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

    print("[INFO] Visualizing result...")
    o3d.visualization.draw_geometries([combined])

    return combined_file

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python apply_transformation_and_combine.py source.pcd target.pcd")
        sys.exit(1)

    source_file = sys.argv[1]
    target_file = sys.argv[2]

    apply_and_combine(source_file, target_file)