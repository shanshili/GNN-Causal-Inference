import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm


class SCMIAlgorithm:
    def __init__(self, data_path, positions_path, delta=50, k_neighbors=5, n_bins=3):
        """
        初始化SCMI算法

        参数:
        data_path: str - 保存数据的.npy文件路径
        positions_path: str - 节点位置文件路径
        delta: float - 距离阈值(米)
        k_neighbors: int - K近邻数量
        n_bins: int - 离散化分箱数量
        """
        self.data = np.load(data_path)  # shape = (301, 1753, 1)
        self.positions = pd.read_csv(positions_path)
        self.delta = delta
        self.k_neighbors = k_neighbors
        self.n_bins = n_bins
        self.num_nodes = self.data.shape[0]
        self.time_steps = self.data.shape[1]

        # 提取节点特征
        self.node_features = self._extract_node_features()

        # 构建图结构
        self.adj_matrix = self._build_graph()

        # 计算空间权重矩阵
        self.weight_matrix = self._compute_spatial_weights()

        # 计算全局Moran's I
        self.moran_i = self._compute_global_morans_i()

    def _extract_node_features(self):
        """(1) 单节点时间序列聚合 - 使用均值作为Com(s_i)"""
        return np.mean(self.data[:, :, 0], axis=1)  # shape = (301,)

    def _build_graph(self):
        """(2) 空间关联图构建 - 使用距离阈值法"""
        # 确保位置数据与节点一一对应
        assert len(self.positions) == self.num_nodes, "位置数据与节点数量不匹配"

        # 计算欧氏距离矩阵
        coords = self.positions[['Longitude', 'Latitude']].values
        distances = squareform(pdist(coords))

        # 距离阈值法构建邻接矩阵
        adj_matrix = (distances <= self.delta).astype(int)

        # 确保对角线为0(节点不与自身相连)
        np.fill_diagonal(adj_matrix, 0)

        return adj_matrix

    def _compute_spatial_weights(self):
        """(2) 空间权重矩阵 - 使用1/d_ij"""
        coords = self.positions[['Longitude', 'Latitude']].values
        distances = squareform(pdist(coords))

        # 将距离转换为空间权重
        weight_matrix = np.zeros_like(distances)
        weight_matrix[distances <= self.delta] = 1 / distances[distances <= self.delta]

        # 确保对角线为0
        np.fill_diagonal(weight_matrix, 0)

        return weight_matrix

    def _compute_global_morans_i(self):
        """(3) 全局Moran's I计算"""
        # 计算均值
        com_mean = np.mean(self.node_features)

        # 计算分子
        numerator = 0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                numerator += self.weight_matrix[i, j] * (self.node_features[i] - com_mean) * (
                            self.node_features[j] - com_mean)

        # 计算分母
        denominator = np.sum((self.node_features - com_mean) ** 2)

        # 计算总权重
        total_weight = np.sum(self.weight_matrix)

        # 计算Moran's I
        moran_i = (self.num_nodes / total_weight) * (numerator / denominator)

        return moran_i

    def _get_conditioning_nodes(self, n_idx, o_idx):
        """(4) 条件节点选择 - 所有与n或o直接相连的节点"""
        connected_to_n = np.where(self.adj_matrix[n_idx] == 1)[0]
        connected_to_o = np.where(self.adj_matrix[o_idx] == 1)[0]

        # 合并并去重
        conditioning_nodes = np.unique(np.concatenate((connected_to_n, connected_to_o)))

        # 排除n和o自身
        conditioning_nodes = conditioning_nodes[~np.isin(conditioning_nodes, [n_idx, o_idx])]

        return conditioning_nodes

    def _discretize_data(self):
        """(4) 离散化 - 使用分箱"""
        # 将数据展平为(节点数*时间步数, 1)
        flat_data = self.data[:, :, 0].reshape(-1, 1)

        # 创建分箱器
        discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal')

        # 应用分箱
        discretized_flat = discretizer.fit_transform(flat_data)

        # 恢复原始形状
        discretized_data = discretized_flat.reshape(self.num_nodes, self.time_steps)

        return discretized_data

    def _compute_conditional_probability(self, x_e_idx, x_n_idx, discretized_data):
        """(4) 空间修正概率计算"""
        # 计算P(x_e)
        e_values = discretized_data[x_e_idx]
        e_counts = np.bincount(e_values.astype(int))
        e_probs = e_counts / len(e_values)

        # 计算P(x_n, x_e)
        n_values = discretized_data[x_n_idx]
        joint_counts = np.zeros((self.n_bins, self.n_bins))

        for t in range(self.time_steps):
            e_bin = int(e_values[t])
            n_bin = int(n_values[t])
            joint_counts[e_bin, n_bin] += 1

        joint_probs = joint_counts / self.time_steps

        # 计算P(x_n|x_e) = P(x_n,x_e)/P(x_e)
        conditional_probs = np.zeros((self.n_bins, self.n_bins))
        for e in range(self.n_bins):
            if e_probs[e] > 0:
                conditional_probs[e] = joint_probs[e] / e_probs[e]

        # 应用空间修正
        spatial_conditional_probs = (1 + self.moran_i) * conditional_probs

        return spatial_conditional_probs, e_probs

    def _compute_joint_distribution(self, x_n_idx, x_o_idx, x_e_idx, discretized_data):
        """(5) 三变量联合分布估计"""
        # 获取三个节点的离散值
        n_values = discretized_data[x_n_idx].astype(int)
        o_values = discretized_data[x_o_idx].astype(int)
        e_values = discretized_data[x_e_idx].astype(int)

        # 初始化联合分布计数
        joint_counts = np.zeros((self.n_bins, self.n_bins, self.n_bins))

        # 计数
        for t in range(self.time_steps):
            n_bin = n_values[t]
            o_bin = o_values[t]
            e_bin = e_values[t]
            joint_counts[e_bin, n_bin, o_bin] += 1

        # 归一化为概率
        joint_probs = joint_counts / self.time_steps

        return joint_probs

    def _compute_scmi(self, x_n_idx, x_o_idx, x_e_idx, discretized_data):
        """(5) SCMI计算"""
        # 获取联合分布
        joint_probs = self._compute_joint_distribution(x_n_idx, x_o_idx, x_e_idx, discretized_data)

        # 计算边缘分布
        e_probs = np.sum(joint_probs, axis=(1, 2))
        n_probs = np.sum(joint_probs, axis=(0, 2))
        o_probs = np.sum(joint_probs, axis=(0, 1))

        # 计算条件概率
        cond_n_e = np.zeros((self.n_bins, self.n_bins))
        cond_o_e = np.zeros((self.n_bins, self.n_bins))

        for e in range(self.n_bins):
            if e_probs[e] > 0:
                cond_n_e[e] = np.sum(joint_probs[e], axis=1) / e_probs[e]
                cond_o_e[e] = np.sum(joint_probs[e], axis=0) / e_probs[e]

        # 计算SCMI
        scmi = 0
        for e in range(self.n_bins):
            for n in range(self.n_bins):
                for o in range(self.n_bins):
                    if joint_probs[e, n, o] > 0:
                        p_neo = joint_probs[e, n, o]
                        p_no_e = joint_probs[e, n, o] / cond_n_e[e, n]
                        p_n_e = cond_n_e[e, n]
                        p_o_e = cond_o_e[e, o]

                        if p_n_e > 0 and p_o_e > 0 and p_no_e > 0:
                            scmi += p_neo * np.log((p_no_e / p_n_e) / p_o_e)

        return scmi

    def compute_causal_graph(self, threshold=None):
        """(6) 因果图生成"""
        # 离散化数据
        discretized_data = self._discretize_data()

        # 初始化因果图
        causal_graph = np.zeros((self.num_nodes, self.num_nodes))

        # 遍历所有节点对
        for n in tqdm(range(self.num_nodes), desc="计算因果图", total=self.num_nodes):
            for o in range(n + 1, self.num_nodes):
                # 检查是否有边连接
                if self.adj_matrix[n, o] == 1:
                    # 获取条件节点
                    conditioning_nodes = self._get_conditioning_nodes(n, o)

                    # 如果没有条件节点，跳过
                    if len(conditioning_nodes) == 0:
                        continue

                    # 计算所有条件节点的SCMI
                    scmi_values = []
                    for e in conditioning_nodes:
                        scmi = self._compute_scmi(n, o, e, discretized_data)
                        scmi_values.append(scmi)

                    # 取最大值作为边的因果强度
                    max_scmi = max(scmi_values)

                    # 如果没有指定阈值，使用最大值的95%分位数
                    if threshold is None:
                        threshold = np.percentile(scmi_values, 95)

                    # 如果SCMI大于阈值，则保留边
                    if max_scmi >= threshold:
                        causal_graph[n, o] = max_scmi
                        causal_graph[o, n] = max_scmi  # 因为是无向图开始，后面会确定方向

        # 确定方向
        directed_graph = np.zeros((self.num_nodes, self.num_nodes))
        for n in range(self.num_nodes):
            for o in range(n + 1, self.num_nodes):
                if causal_graph[n, o] > 0:
                    # 比较方向性
                    n_to_o = causal_graph[n, o]
                    o_to_n = causal_graph[o, n]

                    if n_to_o > o_to_n:
                        directed_graph[n, o] = n_to_o
                    elif o_to_n > n_to_o:
                        directed_graph[o, n] = o_to_n

        return directed_graph


# 使用示例
if __name__ == "__main__":
    # 数据路径
    data_path = 'dataset/sampled_and_standardized_tensor_data_tem.npy'

    # 位置数据路径
    positions_path = 'dataset/TJ_position.csv'

    # 创建SCMI算法实例
    scmi = SCMIAlgorithm(data_path, positions_path)

    # 计算因果图
    causal_graph = scmi.compute_causal_graph()

    # 保存结果
    np.save('causal_graph.npy', causal_graph)

    print("因果图已生成并保存")
