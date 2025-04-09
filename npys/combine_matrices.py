import numpy as np
import argparse
import os

def load_matrix(path):
    try:
        matrix = np.load(path)
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix at '{path}' is not 4x4, got shape {matrix.shape}")
        return matrix
    except Exception as e:
        raise RuntimeError(f"Failed to load matrix from '{path}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Combine two 4x4 transformation matrices from .npy files.")
    parser.add_argument("matrix1", help="Path to the first .npy file (4x4 matrix)")
    parser.add_argument("matrix2", help="Path to the second .npy file (4x4 matrix)")

    args = parser.parse_args()

    # 加载两个矩阵
    mat1 = load_matrix(args.matrix1)
    mat2 = load_matrix(args.matrix2)

    # 组合矩阵
    combined = np.linalg.inv(mat2) @ mat1

    # 固定保存路径
    output_path = "transformation_matrix.npy"
    np.save(output_path, combined)
    print(f"Combined matrix saved to '{output_path}'")

if __name__ == "__main__":
    main()
