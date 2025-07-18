import torch.nn
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv,SAGEConv
import numpy as np
from scipy.special import comb



class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) #edge_index表示邻接矩阵？
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.tanh(x)
        return x



class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.encoder = nn.Sequential(
            nn.Linear(6, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 2),
            nn.LeakyReLU(),
            nn.Linear(2, 1),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.LeakyReLU(),
            nn.Linear(2, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 6),
            nn.Tanh()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded



# 全局读出操作：输出全局特征
class readout(nn.Module):
    def __init__(self, output_size):
        super(readout, self).__init__()
        self.output_size = output_size
        self.weight = torch.nn.Parameter(torch.Tensor(self.output_size, self.output_size))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, h):
        global_h = torch.matmul(torch.mean(h, dim=0), self.weight)
        transformed_global = torch.tanh(global_h)
        sigmoid_scores = torch.sigmoid(torch.mm(h, transformed_global.view(-1, 1)).sum(dim=1))
        h_global = sigmoid_scores.unsqueeze(-1) * h
        return h_global.sum(dim=0).unsqueeze(0)





class AdditiveAttentionReadout(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AdditiveAttentionReadout, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 定义用于计算注意力得分的前馈神经网络
        self.attn = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        # 用于转换全局特征的权重
        self.fc = nn.Linear(input_size, output_size)
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, h):
        """
        :param h: 节点特征，形状为 [num_nodes, input_size]
        :return: 图级别的特征，形状为 [1, output_size]
        """
        num_nodes = h.size(0)
        # 初始化一个隐藏状态作为查询向量
        query = torch.zeros(1, self.hidden_size).to(h.device)
        # 扩展节点特征和查询向量以匹配维度
        h_expanded = h.unsqueeze(1).expand(num_nodes, num_nodes, self.input_size)
        query_expanded = query.expand(num_nodes, self.hidden_size).unsqueeze(0)
        # 计算注意力得分
        attn_scores = self.attn(torch.cat([h_expanded, query_expanded], dim=-1)).squeeze(-1)
        # 使用softmax将得分转换为注意力权重
        attn_weights = F.softmax(attn_scores, dim=1)
        # 计算加权的节点特征
        weighted_h = torch.bmm(attn_weights.unsqueeze(1), h.unsqueeze(0)).squeeze(1)
        # 转换加权后的节点特征以获得最终的图级别特征
        global_feature = self.fc(weighted_h)
        return global_feature


class GraphEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GraphEncoder, self).__init__()
        self.gcn = nn.Linear(input_size, hidden_size)
        self.readout = readout(hidden_size, hidden_size, output_size)

    def forward(self, h, adj):
        h = F.relu(self.gcn(h))
        graph_feature = self.readout(h)
        return graph_feature


class GraphDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GraphDecoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, graph_feature):
        h = F.relu(self.fc1(graph_feature))
        reconstructed_h = self.fc2(h)
        return reconstructed_h


class GraphAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(input_size, hidden_size, output_size)
        self.decoder = GraphDecoder(output_size, hidden_size, input_size)

    def forward(self, h, adj):
        graph_feature = self.encoder(h, adj)
        reconstructed_h = self.decoder(graph_feature)
        return reconstructed_h


class ProjectionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x


class ILGR(nn.Module):
    def __init__(self,  in_channels, hidden_channels, out_channels, num_layers, args):
        super(ILGR,self).__init__()
        torch.manual_seed(1234)
        self.args = args
        # convs 和 attn_convs 分别存储每一层的 SAGEConv 和 GATConv 模块。
        self.convs = torch.nn.ModuleList()
        self.attn_convs = torch.nn.ModuleList()

        # 第一层
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.attn_convs.append(GATConv(hidden_channels, hidden_channels, heads=1, concat=False))

        # 后续层
        for _ in range(1, num_layers):
            self.convs.append(SAGEConv(hidden_channels * 2, hidden_channels))
            self.attn_convs.append(GATConv(hidden_channels * 2, hidden_channels, heads=1, concat=False))

        self.out_layer = torch.nn.Linear(hidden_channels, out_channels)
    def forward(self, x, R_g):
        edge_index = np.array(list(R_g.edges)).T
        # 初始化节点嵌入
        x = torch.cat([x, torch.ones(x.size(0), 1)], dim=-1)

        # 学习节点嵌入
        for i in range(len(self.convs)):
            # 计算邻域嵌入
            h_N = self.attn_convs[i](x, edge_index)

            # 更新节点嵌入
            h_v = self.convs[i](x, edge_index)
            if i > 0:
                h_v = F.relu(torch.cat([x, h_v, h_N], dim=-1))
            else:
                h_v = F.relu(h_v)
            x = h_v

        # 最后一层线性变换
        out = self.out_layer(h_v)
        return out


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(input_dim, hidden_dim, bias=False)
        # self.W = None
        self.q = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self,neighbor_embeddings):
        # self.W = nn.Linear(neighbor_embeddings.size(0), 3, bias=False)
        # print(neighbor_embeddings.size())
        # print(self.W(neighbor_embeddings).size())
        # print(F.relu(self.W(neighbor_embeddings)).size())
        # print(self.q(F.relu(self.W(neighbor_embeddings))).size())
        attention_scores = self.q(F.relu(self.W(neighbor_embeddings)))
        # print('attention_scores',attention_scores)
        attention_weights = F.softmax(attention_scores,dim=0)
        # print('attention_weights', attention_weights.T.size())
        # print('neighbor_embeddings', neighbor_embeddings)
        # print((attention_weights * neighbor_embeddings))
        weighted_neighbor_embeddings = torch.sum(attention_weights * neighbor_embeddings, dim=0)
        # print('weighted_neighbor_embeddings', weighted_neighbor_embeddings.size())
        return weighted_neighbor_embeddings

class NodeEmbeddingModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,args):
        super(NodeEmbeddingModule, self).__init__()
        torch.manual_seed(1234)
        self.args = args
        self.layers = nn.ModuleList([
            nn.Linear((2*input_dim)//(2**i), hidden_dim//(2**i)) for i in range(num_layers)
        ])
        self.attention_layers = nn.ModuleList([
            AttentionLayer(input_dim//(2**i), hidden_dim//(2**i)) for i in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim//(2**(num_layers-1)), output_dim)

    def forward(self, X_v, G):
        h_v = torch.tensor(np.array(X_v),dtype=torch.float32)
        # print('h_v',h_v)# 所有节点的初始特征
        """
        h_v[v] 1*2000 
        neighbor neighbors*2000
        attention_weights  neighbors*1
        h_N_v 邻居特征 1*2000
        [h_v, h_N_v] 1*4000
        h_v() 1*2000
        
        h_v[v] 1*2000 
        neighbor  neighbors*2000
        attention_weights  neighbors*1
        h_N_v 邻居特征 1*2000
        [h_v, h_N_v] 1*4000
        h_v() 1*1000
        neighbor  neighbors*1000
        attention_weights  neighbors*1
        h_N_v 邻居特征 1*1000
        [h_v, h_N_v] 1*2000
        h_v() 1*500
        """
        for l in range(len(self.layers)):   # 当前层
            h_N_v = []    #邻居节点的特征
            for v in range(len(X_v)):   # 遍历当前图中的每个节点
                # print('l', l)
                # print('v',v)
                neighbors = list(G.neighbors(v))   # 获取节点v的邻居
                # print('neighbors',neighbors)
                if not neighbors:  # 孤立节点
                    h_N_v.append(torch.zeros_like(h_v[v]))
                else:
                    neighbor_embeddings = [h_v[i] for i in neighbors]
                    neighbor_embeddings_tensor = torch.stack(neighbor_embeddings, dim=0)
                    # necighbor_embeddings_tensor = torch.tensor(np.array(neighbor_embeddings), dtype=torch.float32)
                    # print(neighbor_embeddings_tensor)
                    # print(h_v[v].unsqueeze(0))
                    h_N_v.append(self.attention_layers[l](neighbor_embeddings_tensor)) # 与上一层的h_v进行attention
            # print('h_N_v', h_N_v)
            h_N_v = torch.stack(h_N_v)
            # print('h_N_v',h_N_v.size())
            # print('h_v', h_v.size())
            # print('.', torch.cat([h_v, h_N_v], dim=1).size())
            # print(self.layers[l])
            h_v = F.relu(self.layers[l](torch.cat([h_v, h_N_v], dim=1)))
        return self.output_layer(h_v[14])



class RegressionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegressionModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//5)
        self.fc3 = nn.Linear(hidden_dim//5, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

class ILGRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,args):
        super(ILGRModel, self).__init__()
        self.embedding_module = NodeEmbeddingModule(input_dim, hidden_dim, output_dim, num_layers,args)
        self.regression_module = RegressionModule(output_dim, output_dim//2, 1)

    def forward(self, X_v, G):
        embeddings = self.embedding_module(X_v, G)
        scores = self.regression_module(embeddings)
        # print('scores',scores)
        return scores

# 定义损失函数
def ranking_loss(scores, true_ranks):
    loss = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            r_ij = true_ranks[i] - true_ranks[j]
            y_hat_ij = scores[i] - scores[j]
            f_r_ij = F.sigmoid(torch.tensor(r_ij,dtype=torch.float32))
            # print('r_ij',r_ij)
            # print('y_hat_ij',y_hat_ij)
            # print('f_r_ij',f_r_ij)
            # print('f_y_hat_ij',F.sigmoid(y_hat_ij))
            # print('log', torch.log(F.sigmoid(y_hat_ij)))
            # print('loss: ', -f_r_ij * torch.log(F.sigmoid(y_hat_ij)) - (1 - f_r_ij) * torch.log(1 - F.sigmoid(y_hat_ij)))
            loss += -f_r_ij * torch.log(F.sigmoid(y_hat_ij)) - (1 - f_r_ij) * torch.log(1 - F.sigmoid(y_hat_ij))

    # print('loss',loss)
    # print('loss', loss/int(comb(len(scores), 2)))
    return loss/int(comb(len(scores), 2))

