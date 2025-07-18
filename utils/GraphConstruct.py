import pandas as pd
import matplotlib.pyplot as plt
from numpy.f2py.cb_rules import cb_map
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable  # 导入 make_axes_locatable

# 全局修改字体
config = {
            "font.family": 'Times New Roman',
            "font.size": 10.5,
            # "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            # "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)

"""
BJ_position = pd.read_csv('../dataset/北京-天津气象数据集2022/北京-天津气象数据集2022/BJ_position.csv')
TJ_position = pd.read_csv('../dataset/北京-天津气象数据集2022/北京-天津气象数据集2022/TJ_position.csv')

dataset_location = pd.concat([BJ_position, TJ_position])
# print(dataset_location)
lat = dataset_location['lat'].values
lon = dataset_location['lon'].values
NO = dataset_location['NO'].values
Label = dataset_location['label'].values
data_location = np.transpose(np.vstack((lat, lon, NO, Label)))
print(data_location)
print("data_location.shape:",data_location.shape)

plt.scatter(lon, lat, 5)  # cmap_name_r, 加_r反转映射关系
plt.ylabel('Latitude')
plt.xlabel('Longitude')
#plt.colorbar()
# plt.savefig('map1.svg', format='svg')
plt.show()
"""


"""
y_pred = DBSCAN(eps = 0.09, min_samples = 5).fit_predict(x)
print(y_pred)
plt.scatter(x[:, 0], x[:, 1],5, c=y_pred)
plt.show()
"""

def location_graph(location):
    test = NearestNeighbors(radius = 0.05)
    test.fit(location)  #?/?????
    """
    dis,ind = test.radius_neighbors(radius = 0.05)
    print("dis:",dis)
    print("ind:",ind)
    """

    # Epsilon neighbor
    #A = test.radius_neighbors_graph(radius = 0.08)
    # K neighbor
    if len(location) < 5:
        A = test.kneighbors_graph(n_neighbors=len(location)-1)
    else:
        A = test.kneighbors_graph(n_neighbors= 4)
    # print("A", A)
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)   # 双向图，但是在这里不影响
    # print("A2:",A)
    location_g = nx.from_numpy_array(A)
    A = nx.to_pandas_adjacency(location_g)
    # nx.draw(location_g,pos = location,with_labels=True, alpha = 0.4, node_size=10, font_size = 5)
    # nx.draw_networkx_nodes(location_g, pos=location, nodelist = [0,1,2], node_size = 20, node_color="r")
    # plt.title('Knn_4_graph')
    # plt.savefig('Knn_4_graph2'+'.svg', format='svg')
    # plt.show()
    return location_g,A

# G = location_graph(data_location[:,(0,1)])

def data_color_graph(data,locationgraph,location,epoch_range):
    #locationgraph = location_graph(location)
    plt.figure()
    plt.title('epoch:'+str(epoch_range))
    nx.draw(locationgraph, pos=location, with_labels=True,  alpha = 0.8, node_size=5,node_color= data, cmap = 'rainbow',font_size = 3)
    plt.savefig('node_quality_epoch'+str(epoch_range)+'.svg', format='svg')


# 绘制—有关图对
def data_color_graph2(score,locationgraph,location,Gpn,color,labels):
    #locationgraph = location_graph(location)
    norm1 = mcolors.Normalize(vmin=np.min(score), vmax=np.max(score))
    fig, ax = plt.subplots()  # 明确创建一个新的图形和轴对象
    plt.title('Graph Pair: '+str(Gpn))
    nx.draw(locationgraph, pos=location, with_labels=labels,  alpha = 0.8, node_size=8,node_color= score, cmap = color,width = 0.6,edge_color = '#BBD6D8',font_size = 0)
    # 设置边框
    ax = plt.gca()  # 获取当前的坐标轴实例
    ax.set_axis_on()  # 打开坐标轴
    ax.spines['bottom'].set_color('0')  # 设置边框颜色
    ax.spines['top'].set_color('0')
    ax.spines['right'].set_color('0')
    ax.spines['left'].set_color('0')
    ax.tick_params(which='both',  width=1, length=1, colors='black')  # 隐藏刻度线
    # plt.colorbar(cm.ScalarMappable(norm=norm1, cmap=color), ax = None)# locationgraph )
    sm = cm.ScalarMappable(cmap=color, norm=norm1)
    sm.set_array([])  # 只是提供一个空数组，因为mappable没有关联的数据
    plt.colorbar(sm, ax=ax)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('similarity score/bar3-Gp'+str(Gpn)+'.svg', format='svg')
    plt.show()

# 绘制—贡献权重图
def data_color_graph3(score,locationgraph,location,color,labels):
    #locationgraph = location_graph(location)
    norm1 = mcolors.Normalize(vmin=np.min(score), vmax=np.max(score))
    fig, ax = plt.subplots()  # 明确创建一个新的图形和轴对象
    plt.title('Contribution weights')
    nx.draw(locationgraph, pos=location, with_labels=labels,  alpha = 0.8, node_size=8,node_color= score, cmap = color,width = 0.8,edge_color = '#9585bd',font_size = 0)
    # 设置边框
    ax = plt.gca()  # 获取当前的坐标轴实例
    ax.set_axis_on()  # 打开坐标轴
    ax.spines['bottom'].set_color('0')  # 设置边框颜色
    ax.spines['top'].set_color('0')
    ax.spines['right'].set_color('0')
    ax.spines['left'].set_color('0')
    ax.tick_params(which='both',  width=1, length=1, colors='black')  # 隐藏刻度线
    # plt.colorbar(cm.ScalarMappable(norm=norm1, cmap=color), ax = None)# locationgraph )
    sm = cm.ScalarMappable(cmap=color, norm=norm1)
    sm.set_array([])  # 只是提供一个空数组，因为mappable没有关联的数据
    plt.colorbar(sm, ax=ax)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('similarity score/bar4.svg', format='svg')
    plt.show()


"""
已知坐标数据 list , 是否是以list的坐标作为节点的标号
如是，那节点标号与坐标数据list行号关联，可以找打对应的NO和Label
No和Label怎么关联对应的数据（因为是乱序）
1. 排序的函数，排序后用行号对应
2. 遍历？ (非常慢，不合理)
关联后，节点权重可以设置为对应的数值
"""
def topological_features_construct(G):
    # print(G.degree())
    topological_features = pd.DataFrame((nx.degree_centrality(G), nx.harmonic_centrality(G), nx.closeness_centrality(G),
                                         nx.betweenness_centrality(G), nx.subgraph_centrality(G), nx.clustering(G)))
    return topological_features
