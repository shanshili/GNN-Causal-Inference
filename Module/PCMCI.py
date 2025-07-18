import sys
import os
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（GNN-Causal-Inference）
project_root = os.path.dirname(current_dir)
# 将项目根目录加入 sys.path
sys.path.append(project_root)
import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt
from tigramite import plotting as tp
from utils.haversine_distance import  compute_haversine_matrix
from tigramite.pcmci_base import PCMCI_base

"""
数据读取
"""
# 构建 dataset 文件夹的路径
dataset_dir = os.path.join(current_dir, '..', 'dataset')
# 确保目标路径存在
os.makedirs(dataset_dir, exist_ok=True)
# 构建 PCMCI 结果保存文件夹路径
output_dir = os.path.join(current_dir, 'PCMCI')
# 如果文件夹不存在，则创建
os.makedirs(output_dir, exist_ok=True)

# 加载数据
position_file = os.path.join(dataset_dir, 'TJ_position.csv')
positions_df = pd.read_csv(position_file)

data_path = os.path.join(dataset_dir, 'sampled_tensor_data_tem.npy')
data = np.load(data_path)  # shape: (301, 1753, 1)

# 去除最后一个维度，使其成为 (301, 1753)
data = data.squeeze(-1)  # shape: (301, 1753)
data = data.T  # shape becomes (1753, 301)

# 定义变量名（可选）
var_names = [f'Node_{i}' for i in range(data.shape[1])]  # 301个节点名

# 创建 DataFrame
dataframe = pp.DataFrame(data, var_names=var_names)

"""
初始化 PCMCI 对象
"""
# 初始化条件独立性检验
cond_ind_test = ParCorr(significance='analytic')
# 初始化 PCMCI 对象
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=1)


# 获取节点坐标（假设列名为 'lat', 'lon'）
positions = positions_df[['lat', 'lon']].values  # shape: (N, 2)
N = positions.shape[0]
# 计算 Haversine 距离矩阵
distance_matrix = compute_haversine_matrix(positions)  # 单位：千米
# 设置距离阈值（根据你的数据调整）
distance_threshold = 10.0  # 例如：只保留 10 km 以内的潜在因果关系

tau_max = 10  # 可根据需要调整
pc_alpha = 0.01  # 显著性阈值

# 手动构造 link_assumptions（等效于 make_link_assumptions）
link_assumptions = {}

for i in range(N):  # 遍历所有变量 i
    for j in range(N):  # 遍历所有变量 j
        for tau in range(1, tau_max + 1):  # lag 从 1 到 tau_max
            if distance_matrix[i, j] > distance_threshold:
                link_assumptions[(j, -tau)] = -1  # 不搜索该连接
            else:
                link_assumptions[(j, -tau)] = 0  # 搜索该连接

results = pcmci.run_pcmciplus(
    tau_max=tau_max,
    pc_alpha=pc_alpha,
    link_assumptions=link_assumptions,
)

# 保存 PCMCI 的结果到 .npz 文件
result_file = os.path.join(output_dir, 'pcmci_results.npz')
np.savez(result_file, **results)
print(f"PCMCI results saved to {result_file}")

# 绘制图结构
graph_fig_file = os.path.join(output_dir, 'causal_graph_pcmci_plus.png')

tp.plot_graph(
    results['graph'],
    val_matrix=results['val_matrix'],
    var_names=var_names,
    link_colorbar_label='MCI',
    node_colorbar_label='auto-MCI',
    link_label_fontsize=10,
    label_fontsize=10,
    tick_label_size=10,
    node_label_size=10,
    edge_ticks=0.5,
    node_ticks=0.5,
    node_size=0.3
)
plt.title("Estimated Causal Graph using PCMCI+", fontsize=12)
plt.tight_layout()
plt.savefig(graph_fig_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Causal graph saved to {graph_fig_file}")