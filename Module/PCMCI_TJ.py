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
from utils.haversine_distance import  compute_haversine_matrix
import networkx as nx
from matplotlib.colors import Normalize
import matplotlib.cm as cm

"""
数据读取
"""
# 构建 dataset 文件夹的路径
dataset_dir = os.path.join(current_dir, '..', 'dataset')
# 确保目标路径存在
os.makedirs(dataset_dir, exist_ok=True)
# 构建 PCMCI 结果保存文件夹路径
output_dir = os.path.join(current_dir, 'PCMCI_TJ')
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
distance_threshold =40.0  # 例如：只保留 _ km 以内的潜在因果关系

tau_max = 16  # 可根据需要调整
pc_alpha = 0.001  # 显著性阈值

# 手动构造 link_assumptions（等效于 make_link_assumptions）
link_assumptions = {}

# 初始化统计变量
k_counts = {}  # 每个节点 i 的有链接边数（考虑时间）
d_counts = {}  # 每个节点 i 的有链接的源点数（不考虑时间）

for i in range(N):  # i 是目标变量（被影响变量）
    k = 0
    d_set = set()  # 用于记录不重复的源节点 j
    link_assumptions[i] = {}
    for j in range(N):  # j 是源变量（影响变量）
        for tau in range(1, tau_max + 1):
            if distance_matrix[i, j] > distance_threshold:
                # 不搜索该连接 → 不添加该键值对即可
                continue
            else:
                # 允许连接，并假设方向为 'o-o'
                link_assumptions[i][(j, -tau)] = 'o-o'
                k += 1
                d_set.add(j)  # 将源节点 j 加入集合（自动去重）
    # 保存统计结果
    k_counts[i] = k
    d_counts[i] = len(d_set)
    print(f"Node {i}: 共有 {k_counts[i]} 条链接（含时间），来自 {d_counts[i]} 个不同节点")


results = pcmci.run_pcmciplus(
    tau_max=tau_max,
    pc_alpha=pc_alpha,
    link_assumptions=link_assumptions,
)

# 参数格式化，用于文件名
tau_str = f'tau{tau_max}'
alpha_str = f'alpha{pc_alpha:.3f}'.replace('.', 'p')  # 0.001 → alpha0p001
dist_str = f'dist{distance_threshold:.1f}km'.replace('.', 'd')  # 5.0 → dist5d0km
# 保存 PCMCI 的结果到 .npz 文件
filename3 = f'pcmci_results_{tau_str}_{alpha_str}_{dist_str}.npz'
result_file = os.path.join(output_dir, filename3)
np.savez(result_file, **results)
print(f"PCMCI results saved to {result_file}")

# 绘制图结构
# graph_fig_file = os.path.join(output_dir, 'causal_graph_pcmci_plus.png')
#
# tp.plot_graph(
#     results['graph'],
#     val_matrix=results['val_matrix'],
#     var_names=var_names,
#     link_colorbar_label='MCI',
#     node_colorbar_label='auto-MCI',
#     link_label_fontsize=5,
#     label_fontsize=6,
#     tick_label_size=6,
#     node_label_size=6,
#     edge_ticks=0.1,
#     node_ticks=0.1,
#     node_size=0.1
# )
# plt.title("Estimated Causal Graph using PCMCI+", fontsize=12)
# plt.tight_layout()
# plt.savefig(graph_fig_file, dpi=600, bbox_inches='tight')
# plt.close()

# print(f"Causal graph saved to {graph_fig_file}")



graph = results['graph']

# 统计每个节点的自关联数量（节点到自己的因果联系）
node_self_counts = np.zeros(N)
for i in range(N):
    for tau in range(1, tau_max + 1):
        if graph[i, i, tau] == '-->':  # 检查节点i到节点i的关联
            node_self_counts[i] += 1

# 构建连接统计字典（排除自关联）
edge_counts = {}
for i in range(N):
    print(f"节点{i}")
    for j in range(N):
        if i == j:  # 排除自关联
            continue
        count = 0
        for tau in range(1, tau_max + 1):
            if graph[i, j, tau] == '-->':
                count += 1
        if count > 0:
            edge_counts[(i, j)] = count
            print(f"与{j}链接{count}")


"""
因果图
"""
# 创建 NetworkX 图对象
G = nx.DiGraph()

# 添加节点（仅有效位置），记录自关联数量
for i in range(N):
    G.add_node(i, pos=(positions[i, 0], positions[i, 1]),self_count=node_self_counts[i])

# 添加边（仅两个节点都有效的边），排除自关联
for (i, j), count in edge_counts.items():
    if  i != j:  # 确保不是自关联
        G.add_edge(j, i, weight=count)  # 从j到i的有向边

# 获取节点位置
pos = nx.get_node_attributes(G, 'pos')
# 打印统计信息
# 打印统计信息
print("=== 节点自关联、入边、出边统计 ===")
total_tau = sum(edge_counts.values())
# 创建统计表
stats = []
outcoming_node_count = []
for i in G.nodes:
    # 自关联数量
    self_count = G.nodes[i]['self_count']
    # 入边统计
    incoming_edges = [u for u, v in G.edges if v == i]  # 指向该节点的源节点列表
    incoming_edge_weights = [G[u][v]['weight'] for u, v in G.edges if v == i]
    total_incoming_tau = sum(incoming_edge_weights)
    # 出边统计
    outgoing_edges = [v for u, v in G.edges if u == i]  # 该节点指向的目标节点列表
    outgoing_edge_weights = [G[i][v]['weight'] for v in outgoing_edges]
    total_outgoing_tau = sum(outgoing_edge_weights)

    outcoming_node_count.append(len(outgoing_edges))

    # 保存统计
    stats.append({
        'Node': i,
        'SelfLinkCount': self_count,
        'IncomingNodeCount': len(incoming_edges),
        'TotalIncomingTau': total_incoming_tau,
        'OutgoingNodeCount': len(outgoing_edges),
        'TotalOutgoingTau': total_outgoing_tau
    })

    # 打印
    print(f"Node {i}:")
    print(f"  自关联数量: {self_count}")
    print(f"  指向它的节点数: {len(incoming_edges)}")
    print(f"  入边因果联系总数（τ）: {total_incoming_tau}")
    print(f"  被它指向的节点数: {len(outgoing_edges)}")
    print(f"  出边因果联系总数（τ）: {total_outgoing_tau}")
    print("-" * 40)

print("=============================")

# 设置绘图参数
fig, ax = plt.subplots(figsize=(12, 10), dpi=600)

# 设置节点的颜色映射（基于“指向它的节点数”）
if outcoming_node_count:
    max_incoming = max(outcoming_node_count)
    node_norm = Normalize(vmin=0, vmax=max_incoming)
    node_cmap = cm.plasma_r
else:
    node_norm = Normalize(vmin=0, vmax=1)
    node_cmap = cm.plasma_r

# 获取节点颜色
node_colors = [node_cmap(node_norm(c)) for c in outcoming_node_count]

# 获取边的权重（因果联系数量）
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

# 设置边的颜色映射（基于入边因果联系总数）
if edge_weights:
    max_edge_weight = max(edge_weights)
    edge_norm = Normalize(vmin=0, vmax=max_edge_weight)
    edge_cmap = cm.plasma_r
else:
    edge_norm = Normalize(vmin=0, vmax=1)
    edge_cmap = cm.plasma_r

# 边颜色
edge_colors = [edge_cmap(edge_norm(w)) for w in edge_weights]


# ✅ 绘制节点（颜色表示自关联数量）
node_colors = [node_cmap(node_norm(c)) for c in outcoming_node_count]
nodes = nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, alpha=0.8, ax=ax)

# ✅ 绘制边（颜色表示因果联系数量）
edge_colors = [edge_cmap(edge_norm(w)) for w in edge_weights]
edges = nx.draw_networkx_edges(
    G,
    pos,
    edge_color=[edge_cmap(edge_norm(w)) for w in edge_weights],
    width=0.8,
    alpha=0.8,
    arrows=True,
    #arrowstyle='simple, head_length=0.2, head_width=0.3, tail_width=0.05',
    node_size=50,
    arrowstyle="->",
    arrowsize=5,
    connectionstyle='arc3,rad=0.2',
    ax=ax
)

# ✅ 添加颜色条 - 边因果联系数量
sm_edges = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
sm_edges.set_array([])
cbar_edges = fig.colorbar(sm_edges, ax=ax, orientation='vertical', shrink=0.3, pad=0.05)
cbar_edges.set_label('Edge Causal Link Count')

# ✅ 添加颜色条 - 节点出边数量
sm_nodes = plt.cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
sm_nodes.set_array([])
cbar_nodes = fig.colorbar(sm_nodes, ax=ax, orientation='vertical', shrink=0.3, pad=0.05)
cbar_nodes.set_label('Number of Outgoing Nodes')

# 可选：添加节点编号
# labels = {i: str(i) for i in G.nodes}
# nx.draw_networkx_labels(G, pos, labels, font_size=6, font_color='black', ax=ax)

# 设置标题和坐标轴
ax.set_title("Causal Graph")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle='--', alpha=0.3)
plt.axis('on')
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.tight_layout()
# 保存图像
# 拼接文件名
filename1 = f'network_causal_graph_{tau_str}_{alpha_str}_{dist_str}.png'

# 保存图像
plt.savefig(os.path.join(output_dir, filename1), dpi=600, bbox_inches='tight')
print(f"因果图已保存至：{os.path.join(output_dir, filename1)}")

"""
因果矩阵热力图
"""
# 创建 N×N 矩阵
N = positions.shape[0]
causal_matrix = np.zeros((N, N))

# 填充矩阵（i → j 的因果联系数量）
for (i, j), count in edge_counts.items():
    causal_matrix[i, j] = count  # 注意：G.add_edge(j, i) 表示 j ← i，即 i → j
# 添加自关联（i → i）
for i in range(N):
    causal_matrix[i, i] = node_self_counts[i]  # node_self_counts[i] 是 i → i 的因果联系数量

# 创建绘图对象
fig_matrix, ax_matrix = plt.subplots(figsize=(10, 8), dpi=1800)

# 绘制热力图
cax = ax_matrix.imshow(causal_matrix, cmap=cm.plasma_r, interpolation='none', aspect='auto')

# 添加颜色条
cbar = fig_matrix.colorbar(cax, ax=ax_matrix)
cbar.set_label('Causal Link Count (τ)')

# 设置标题和坐标轴
ax_matrix.set_title("Node Causal Link Count Matrix (i → j)", fontsize=14)
ax_matrix.set_xlabel("Target Node (j)", fontsize=12)
ax_matrix.set_ylabel("Source Node (i)", fontsize=12)

# 设置坐标轴刻度
ax_matrix.set_xticks(np.arange(N))
ax_matrix.set_yticks(np.arange(N))
ax_matrix.tick_params(axis='both', which='major', labelsize=1)

plt.tight_layout()

filename2 = f'causal_matrix_{tau_str}_{alpha_str}_{dist_str}.png'
plt.savefig(os.path.join(output_dir, filename2), dpi=1800, bbox_inches='tight')
print(f"因果矩阵热力图已保存至：{os.path.join(output_dir, filename2)}")

