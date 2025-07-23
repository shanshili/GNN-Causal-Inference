import numpy as np


def compute_euclidean_matrix(positions, scale=1.0):
    """
    计算欧氏距离矩阵（平面距离）

    参数:
        positions: 节点坐标数组，shape (N, 2) 或 (N, 3)
        scale: 单位缩放系数（例如：1=米，0.001=千米）

    返回:
        distance_matrix: 距离矩阵 (N, N)，单位与输入相同
    """
    N = positions.shape[0]
    distance_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            diff = positions[i] - positions[j]
            distance_matrix[i, j] = np.sqrt(np.sum(diff ** 2)) * scale

    return distance_matrix
