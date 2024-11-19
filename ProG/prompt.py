import faiss
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv, BatchNorm
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_adj, coalesce
from sklearn.cluster import KMeans
import numpy as np
from .utils import act
import warnings
from deprecated.sphinx import deprecated


# the original GNN which no pooling
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, pool=None, gnn_type='GAT'):
        super().__init__()

        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.gnn_type = gnn_type
        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim

        if gcn_layer_num < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        elif gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(gcn_layer_num - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        if pool is None:
            self.pool = global_mean_pool
        else:
            self.pool = pool

    def forward(self, x, edge_index, batch, coord=None):
        for conv in self.conv_layers[0:-1]:
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, training=self.training)

        node_emb = self.conv_layers[-1](x, edge_index)
        graph_emb = self.pool(node_emb, batch.long())
        return graph_emb


# use kmeans to cluster and pool
class ClusterGNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, cluster_sizes=[100, 50, 10], pool=None, gnn_type='GAT'):
        super().__init__()

        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.cluster_sizes = cluster_sizes
        self.gnn_type = gnn_type
        self.conv_layers = torch.nn.ModuleList()

        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        # if out_dim is None:
        #     out_dim = hid_dim

        # if gcn_layer_num < 2:
        #     raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        # elif gcn_layer_num == 2:
        #     self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        # else:
        #     layers = [GraphConv(input_dim, hid_dim)]
        #     for i in range(gcn_layer_num - 2):
        #         layers.append(GraphConv(hid_dim, hid_dim))
        #     layers.append(GraphConv(hid_dim, out_dim))
        #     self.conv_layers = torch.nn.ModuleList(layers)
        for _ in range(len(cluster_sizes) + 1):  # +1 for the initial layer before clustering
            self.conv_layers.append(GraphConv(input_dim, hid_dim))
            input_dim = hid_dim

        if pool is None:
            self.pool = global_mean_pool
        else:
            self.pool = pool

    def forward(self, x, edge_index, batch, coord=None):
        cluster_results = [coord]
        # for conv in self.conv_layers[0:-1]:
        #     x = conv(x, edge_index)
        #     x = act(x)
        #     x = F.dropout(x, training=self.training)

        for i, conv in enumerate(self.conv_layers[0:-1]):
            x = conv(x, edge_index.to(x.device))
            x = act(x)
            x = F.dropout(x, training=self.training)
            if i < len(self.cluster_sizes):  # Check if it's a layer before clustering
                x, edge_index, batch, new_coords = self.cluster_and_pool(x, edge_index, batch, cluster_results[i],
                                                                         self.cluster_sizes[i])
                # x, edge_index, batch, new_coords = self.cluster_and_pool_faiss(x, edge_index, batch, cluster_results[i],
                #                                                          self.cluster_sizes[i])
                cluster_results.append(new_coords)

        node_emb = self.conv_layers[-1](x, edge_index.to(x.device))
        graph_emb = self.pool(node_emb, batch.long())
        return graph_emb

    # use original kmeans
    def cluster_and_pool(self, x, edge_index, batch, coord, cluster_size):
        device = x.device
        # Implement clustering based on 'coord' and update 'x', 'edge_index', 'batch' accordingly.
        batch_size = batch.max().item() + 1
        total_clusters = 0  # Track total number of clusters across all batches
        cluster_labels_global = torch.empty(coord.size(0), dtype=torch.long, device=coord.device)
        new_coords_list = []
        for i in range(batch_size):
            # for each sample in the batch
            mask = (batch == i)
            batch_coord = coord[mask]

            current_cluster_size = min(cluster_size, batch_coord.size(0))
            if current_cluster_size == batch_coord.size(0):
                batch_labels = torch.arange(current_cluster_size, dtype=torch.long,
                                            device=coord.device) + total_clusters
                new_coords = batch_coord
            else:
                kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(batch_coord.detach().cpu().numpy())
                # batch_labels = torch.tensor(kmeans.labels_, dtype=torch.long, device=coord.device) + total_clusters
                # new_coords = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=coord.device)
                batch_labels = torch.from_numpy(kmeans.labels_).to(dtype=torch.long, device=coord.device) + total_clusters
                new_coords = torch.from_numpy(kmeans.cluster_centers_).to(dtype=torch.float, device=coord.device)

            # cluster_labels_global[mask] = batch_labels + cluster_size * i
            cluster_labels_global[mask] = batch_labels
            total_clusters += current_cluster_size
            new_coords_list.append(new_coords)

        new_coords = torch.cat(new_coords_list, dim=0)

        new_x, new_edge_index, new_batch = self.update_graph(x, edge_index, batch, cluster_labels_global,
                                                             total_clusters)
        return new_x, new_edge_index, new_batch, new_coords


    # use faiss for kmeans, even slower than the original kmeans, obsoleted
    # def cluster_and_pool_faiss(self, x, edge_index, batch, coord, cluster_size):
    #     device = x.device
    #     batch_size = batch.max().item() + 1
    #     total_clusters = 0
    #     cluster_labels_global = torch.empty(coord.size(0), dtype=torch.long, device=device)
    #     new_coords_list = []
    #
    #     for i in range(batch_size):
    #         # # for gpu
    #         # mask = (batch == i)
    #         # batch_coord = coord[mask].cpu().detach().numpy()
    #         #
    #         # clus = faiss.Clustering(batch_coord.shape[1], cluster_size)
    #         # clus.verbose = False
    #         # clus.niter = 20
    #         # clus.nredo = 5
    #         # res = faiss.StandardGpuResources()
    #         # flat_config = faiss.GpuIndexFlatConfig()
    #         # flat_config.device = 0  # GPU number
    #         # index = faiss.GpuIndexFlatL2(res, batch_coord.shape[1], flat_config)
    #         # # batch_coord = batch_coord.cpu().detach().numpy()
    #         # clus.train(batch_coord, index)
    #         # _, I = index.search(batch_coord, 1)
    #         # batch_labels = torch.from_numpy(I.squeeze()).to(device) + total_clusters
    #         #
    #         # centroids = faiss.vector_float_to_array(clus.centroids).reshape(cluster_size, batch_coord.shape[1])
    #         # new_coords = torch.from_numpy(centroids).float().to(device)
    #         #
    #         # cluster_labels_global[mask] = batch_labels
    #         # total_clusters += cluster_size
    #         # new_coords_list.append(new_coords)
    #
    #         # for cpu
    #         mask = (batch == i)
    #         batch_coord = coord[mask].cpu().detach().numpy()
    #
    #         index = faiss.IndexFlatL2(batch_coord.shape[1])
    #         clus = faiss.Clustering(batch_coord.shape[1], cluster_size)
    #         clus.verbose = False
    #         clus.niter = 20
    #         clus.nredo = 5
    #
    #         clus.train(batch_coord, index)
    #         _, I = index.search(batch_coord, 1)
    #         batch_labels = torch.from_numpy(I.squeeze()).to(device) + total_clusters
    #
    #         centroids = faiss.vector_float_to_array(clus.centroids).reshape(cluster_size, batch_coord.shape[1])
    #         new_coords = torch.from_numpy(centroids).float().to(device)
    #
    #         cluster_labels_global[mask] = batch_labels
    #         total_clusters += cluster_size
    #         new_coords_list.append(new_coords)
    #
    #     new_coords = torch.cat(new_coords_list, dim=0)
    #     new_x, new_edge_index, new_batch = self.update_graph(x, edge_index, batch, cluster_labels_global,
    #                                                          total_clusters)
    #     return new_x, new_edge_index, new_batch, new_coords


    def update_graph(self, x, edge_index, batch, cluster_labels, total_clusters):
        device = x.device
        # Initialize the new features tensor
        new_x = torch.zeros((total_clusters, x.size(1)), dtype=torch.float, device=device)

        for i in range(total_clusters):
            mask = (cluster_labels == i)
            new_x[i] = torch.mean(x[mask], dim=0) if mask.any() else torch.zeros((x.size(1),), dtype=torch.float,
                                                                                 device=device)

        # Map the original node indices to their respective cluster indices
        cluster_mapping = cluster_labels
        edge_index_mapped = cluster_mapping[edge_index]

        # # Create a new edge_index with no self-loops and duplicates for clusters
        # new_edge_index, _ = coalesce(edge_index_mapped, None, total_clusters, total_clusters, op='min')

        # Remove self-loops
        edge_index_mapped, _ = torch_geometric.utils.remove_self_loops(edge_index_mapped)
        # Remove duplicate edges
        new_edge_index, _ = torch_geometric.utils.coalesce(edge_index_mapped, None, total_clusters, total_clusters)

        # Update batch information for each cluster
        new_batch = torch.zeros(total_clusters, dtype=torch.long, device=device)
        for i in range(total_clusters):
            mask = (cluster_labels == i)
            new_batch[i] = batch[mask][0] if mask.any() else -1  # -1 for empty clusters if any

        return new_x, new_edge_index, new_batch


class GNNClusterIDPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, cluster_size, gnn_type='GCN'):
        super(GNNClusterIDPredictor, self).__init__()

        if gnn_type == 'GCN':
            GNNLayer = GCNConv
        elif gnn_type == 'GAT':
            GNNLayer = GATConv
        elif gnn_type == 'TransformerConv':
            GNNLayer = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN, and TransformerConv')

        self.gnn1 = GNNLayer(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.gnn2 = GNNLayer(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        # 如果需要，可以添加更多的GNN层
        self.output_layer = nn.Linear(hidden_dim, cluster_size)
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, edge_index):
        # normalize to 0-1
        x = F.normalize(x, p=2)
        x = F.relu(self.bn1(self.gnn1(x, edge_index)))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn2(self.gnn2(x, edge_index)))
        x = self.output_layer(x)
        return F.softmax(x, dim=1)


# use gnn to learn the cluster id for pooling
class SoftClusterGNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, cluster_sizes=[100, 50, 10], pool=None, gnn_type='GAT', phase='train'):
        super().__init__()

        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.cluster_sizes = cluster_sizes
        self.gnn_type = gnn_type
        self.phase = phase
        self.conv_layers = torch.nn.ModuleList()
        self.cluster_id_predictor = torch.nn.ModuleList()

        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"

        for idx in range(len(cluster_sizes) + 1):  # +1 for the initial layer before clustering
            self.conv_layers.append(GraphConv(input_dim, hid_dim))
            input_dim = hid_dim

        for idx in range(len(cluster_sizes)):
            # # input_dim, hid_dim = 256, 256
            # self.cluster_id_predictor.append(GNNClusterIDPredictor(input_dim, hid_dim, self.cluster_sizes[idx]))
            # input_dim = hid_dim

            # input_dim, hid_dim = 2, 2
            self.cluster_id_predictor.append(GNNClusterIDPredictor(2, int(cluster_sizes[idx]/2), self.cluster_sizes[idx]))

        if pool is None:
            self.pool = global_mean_pool
        else:
            self.pool = pool

        # freeze the params
        if self.phase in ['train', 'finetune']:
            pass
        else:
            for predictor in self.cluster_id_predictor:
                for p in predictor.parameters():
                    p.requires_grad = False
            for conv in self.conv_layers:
                for p in conv.parameters():
                    p.requires_grad = False


    def learn_cluster_ids(self, x, edge_index, batch, layer_idx):
        # 确保层索引在合理范围内
        if layer_idx < 0 or layer_idx >= len(self.cluster_id_predictor):
            raise IndexError("layer_idx is out of bounds")

        assignment_probs_list = []
        cluster_size = self.cluster_id_predictor[layer_idx].output_layer.out_features
        # try:
        #     print(batch.max().item())
        # except Exception as e:
        #     print(e)
        for i in range(batch.max().item() + 1):  # 遍历每个批次中的样本
            # 获取当前样本的节点和边
            mask = (batch == i)
            x_i = x[mask]
            edge_index_i = edge_index[:, mask[edge_index[0]] & mask[edge_index[1]]]

            if x_i.size(0) == 0 or edge_index_i.numel() == 0:
                size = max([len(x[(batch==idx)]) for idx in range(batch.max().item() + 1)])
                assignment_probs_i = torch.zeros((size, cluster_size), device=x.device, dtype=x.dtype)
            else:
                edge_index_i = edge_index_i - edge_index_i.min(dim=1, keepdim=True).values

                # 调整edge_index_i以防止索引超出范围
                # _, edge_index_i = torch_geometric.utils.subgraph(i, edge_index, relabel_nodes=True)

                # 使用对应层的聚类ID预测器
                predictor = self.cluster_id_predictor[layer_idx]
                assignment_probs_i = predictor(x_i, edge_index_i)

            assignment_probs_list.append(assignment_probs_i)

        # 拼接所有样本的聚类概率
        assignment_probs = torch.cat(assignment_probs_list, dim=0)

        return assignment_probs

    def forward(self, x, edge_index, batch, coord=None):
        cluster_results = [coord] # use the coord for cluster
        vertex_features = [] # record the vertex features for each hierarchical layer

        for i, conv in enumerate(self.conv_layers[0:-1]):
            x = conv(x, edge_index.to(x.device))
            # print("x = conv(x, edge_index.to(x.device))")
            x = act(x)
            x = F.dropout(x, training=self.training)
            if i < len(self.cluster_sizes):  # Check if it's a layer before clustering
                # x, edge_index, batch, new_coords = self.cluster_and_pool(x, edge_index, batch, cluster_results[i],
                #                                                          self.cluster_sizes[i])
                assignment_probs = self.learn_cluster_ids(cluster_results[i].to(x.device), edge_index, batch, i)

                if batch.shape[0] != assignment_probs.shape[0]:
                    pass

                x, edge_index, batch, new_coords = self.cluster_and_pool(x, edge_index, batch, assignment_probs,
                                                                                 self.cluster_sizes[i], cluster_results[i])
                cluster_results.append(new_coords)
                vertex_features.append(x)

        node_emb = self.conv_layers[-1](x, edge_index.to(x.device))
        graph_emb = self.pool(node_emb, batch.long())

        if self.phase == 'train':
            return graph_emb
        else:
            return graph_emb, vertex_features, cluster_results

    # use kmeans
    def cluster_and_pool(self, x, edge_index, batch, assignment_probs, cluster_size, coord):
        device = x.device
        new_x_list = []
        new_edge_index_list = []
        new_coords_list = []
        new_batch_list = []
        total_clusters = 0  # 总聚类数将基于每个样本动态计算

        for i in range(batch.max().item() + 1):  # 遍历每个批次中的样本
            # 选择当前样本的节点
            mask = batch == i
            # try:
            x_i = x[mask]
            # except Exception:
            #     print('a')
            #     pass
            coord = coord.to(mask.device)
            coord_i = coord[mask]

            if mask.shape[0] != assignment_probs.shape[0]:
                pass
                raise ValueError(f"The shape of the mask {mask.shape} at index 0 does not match the shape of the indexed tensor {assignment_probs.shape} at index 0.")

            assignment_probs_i = assignment_probs[mask]

            # 计算当前样本的新节点特征
            new_x_i = torch.matmul(assignment_probs_i.t(), x_i)  # [C, node_feature_dim]
            new_x_list.append(new_x_i)

            # 提取子图并重新标记节点
            node_idx = mask.nonzero(as_tuple=False).view(-1)
            subgraph_mask = torch.zeros(x.size(0), dtype=torch.bool, device=device)
            subgraph_mask[node_idx] = True
            edge_index_i, _ = torch_geometric.utils.subgraph(subgraph_mask, edge_index, relabel_nodes=True)

            # # 获取当前样本的边索引
            # edge_index_i = edge_index[:, mask[edge_index[0]] & mask[edge_index[1]]]
            # _, edge_index_i = torch_geometric.utils.subgraph(mask, edge_index, relabel_nodes=True)

            # 计算当前样本的新邻接矩阵
            cluster_index_i = torch.argmax(assignment_probs_i, dim=1)  # 获取最可能的聚类ID
            source_clusters = cluster_index_i[edge_index_i[0, :]]
            target_clusters = cluster_index_i[edge_index_i[1, :]]
            new_edge_index_i = torch.stack([source_clusters, target_clusters], dim=0)

            # 移除自环和重复边
            new_edge_index_i, _ = torch_geometric.utils.remove_self_loops(new_edge_index_i)
            edge_result = torch_geometric.utils.coalesce(new_edge_index_i, None, cluster_size, cluster_size)
            if isinstance(edge_result, tuple):
                # some versions
                new_edge_index_i, _ = edge_result
            else:
                # other versions
                new_edge_index_i = edge_result
            # new_edge_index_i, _ = torch_geometric.utils.coalesce(new_edge_index_i, None, cluster_size, cluster_size)
            # new_edge_index_list.append(new_edge_index_i + total_clusters)  # 调整索引基于总聚类数
            new_edge_index_list.append(new_edge_index_i + total_clusters)  # 调整索引基于总聚类数

            # 计算新的聚类中心坐标
            for cluster_id in range(cluster_size):
                cluster_mask = (cluster_index_i == cluster_id)
                if cluster_mask.any():
                    cluster_coords = coord_i[cluster_mask]
                    # cal the center of cluster
                    cluster_center = cluster_coords.mean(dim=0)
                    # cluster_prob = assignment_probs_i[cluster_mask, cluster_id].unsqueeze(1)
                    # weighted_coords = cluster_coords * cluster_prob.to(cluster_coords.device)
                    # cluster_center = weighted_coords.sum(dim=0) / cluster_prob.sum().to(weighted_coords.device)
                    new_coords_list.append(cluster_center)
                else:
                    new_coords_list.append(torch.zeros((coord_i.size(1))).to('cuda'))

            # 更新总聚类数
            total_clusters += cluster_size

            # 更新当前样本的批信息
            new_batch_i = torch.full((cluster_size,), i, dtype=torch.long, device=device)
            new_batch_list.append(new_batch_i)

        # 合并所有样本的结果
        new_x = torch.cat(new_x_list, dim=0)
        new_edge_index = torch.cat(new_edge_index_list, dim=1)
        new_batch = torch.cat(new_batch_list, dim=0)
        new_coords = torch.stack(new_coords_list)

        return new_x, new_edge_index, new_batch, new_coords

    def update_graph(self, x, edge_index, batch, cluster_labels, total_clusters):
        device = x.device
        # Initialize the new features tensor
        new_x = torch.zeros((total_clusters, x.size(1)), dtype=torch.float, device=device)

        for i in range(total_clusters):
            mask = (cluster_labels == i)
            new_x[i] = torch.mean(x[mask], dim=0) if mask.any() else torch.zeros((x.size(1),), dtype=torch.float,
                                                                                 device=device)

        # Map the original node indices to their respective cluster indices
        cluster_mapping = cluster_labels
        edge_index_mapped = cluster_mapping[edge_index]

        # # Create a new edge_index with no self-loops and duplicates for clusters
        # new_edge_index, _ = coalesce(edge_index_mapped, None, total_clusters, total_clusters, op='min')

        # Remove self-loops
        edge_index_mapped, _ = torch_geometric.utils.remove_self_loops(edge_index_mapped)
        # Remove duplicate edges
        new_edge_index, _ = torch_geometric.utils.coalesce(edge_index_mapped, None, total_clusters, total_clusters)

        # Update batch information for each cluster
        new_batch = torch.zeros(total_clusters, dtype=torch.long, device=device)
        for i in range(total_clusters):
            mask = (cluster_labels == i)
            new_batch[i] = batch[mask][0] if mask.any() else -1  # -1 for empty clusters if any

        return new_x, new_edge_index, new_batch


class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None):
        """
        :param token_dim:
        :param token_num_per_group:
        :param group_num:   the total token number = token_num_per_group*group_num, in most cases, we let group_num=1.
                            In prompt_w_o_h mode for classification, we can let each class correspond to one group.
                            You can also assign each group as a prompt batch in some cases.

        :param prune_thre: if inner_prune is None, then all inner and cross prune will adopt this prune_thre
        :param isolate_tokens: if Trure, then inner tokens have no connection.
        :param inner_prune: if inner_prune is not None, then cross prune adopt prune_thre whereas inner prune adopt inner_prune
        """
        super(LightPrompt, self).__init__()

        self.inner_prune = inner_prune

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])

        self.token_init(init_method="kaiming_uniform")

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self, ):
        """
        each token group is viewed as a prompt sub-graph.
        turn the all groups of tokens as a batch of prompt graphs.
        :return:
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch


class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.01):
        super(HeavyPrompt, self).__init__(token_dim, token_num, 1, inner_prune)  # only has one prompt graph.
        self.cross_prune = cross_prune

    def forward(self, graph_batch: Batch):
        """
        TODO: although it recieves graph batch, currently we only implement one-by-one computing instead of batch computing
        TODO: we will implement batch computing once we figure out the memory sharing mechanism within PyG
        :param graph_batch:
        :return:
        """
        # device = torch.device("cuda")
        # device = torch.device("cpu")

        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num
            # pg_x = pg.x.to(device)
            # g_x = g.x.to(device)
            
            cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
            
            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num
            
            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch

class FrontAndHead(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3):

        super().__init__()

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch, gnn):
        prompted_graph = self.PG(graph_batch)
        graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        pre = self.answering(graph_emb)

        return pre


@deprecated(version='1.0', reason="Pipeline is deprecated, use FrontAndHead instead")
class Pipeline(torch.nn.Module):
    def __init__(self, input_dim, dataname, gcn_layer_num=2, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3, gnn_type='TransformerConv'):
        warnings.warn("deprecated", DeprecationWarning)

        super().__init__()
        # load pre-trained GNN
        self.gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=gcn_layer_num, gnn_type=gnn_type)
        pre_train_path = './pre_trained_gnn/{}.GraphCL.{}.pth'.format(dataname, gnn_type)
        self.gnn.load_state_dict(torch.load(pre_train_path))
        print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
        for p in self.gnn.parameters():
            p.requires_grad = False

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch: Batch):
        prompted_graph = self.PG(graph_batch)
        graph_emb = self.gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        pre = self.answering(graph_emb)

        return pre



if __name__ == '__main__':
    pass
