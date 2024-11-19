import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(coordinates, edge_index, filename=None):
    G = nx.Graph()

    # 添加顶点
    for i, coord in enumerate(coordinates):
        G.add_node(i, pos=tuple(coord))

    # 添加边
    edges = zip(edge_index[0], edge_index[1])
    G.add_edges_from(edges)

    # 绘制图
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, ax=ax, with_labels=False, node_color='skyblue', node_size=50,
            edge_color='k', linewidths=1, font_size=15,
            arrows=False)
    ax.set_title('Visualized Graph', fontsize=20)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, format='svg', bbox_inches='tight')
        plt.close()

def graph_views_ori(data, aug='random', aug_ratio=0.1):
    if isinstance(aug, list):
        aug = aug[0]

    if aug == 'dropN':
        data = drop_nodes_ori(data, aug_ratio)
    elif aug == 'permE':
        data = permute_edges_ori(data, aug_ratio)
    elif aug == 'maskN':
        data = mask_nodes_ori(data, aug_ratio)
    elif aug == 'random':
        n = np.random.randint(2)
        if n == 0:
            data = drop_nodes_ori(data, aug_ratio)
        elif n == 1:
            data = permute_edges_ori(data, aug_ratio)
        else:
            print('augmentation error')
            assert False
    return data

def graph_views(data, aug='random', aug_ratio=0.1, coordinates=None):
    if isinstance(aug, list):
        aug = aug[0]

    if aug == 'dropN':
        data, coordinates = drop_nodes(data, aug_ratio, coordinates)
    elif aug == 'permE':
        data = permute_edges(data, aug_ratio)
    elif aug == 'maskN':
        data = mask_nodes(data, aug_ratio)
    elif aug == 'random':
        n = np.random.randint(2)
        if n == 0:
            data, coordinates = drop_nodes(data, aug_ratio, coordinates)
        elif n == 1:
            data = permute_edges(data, aug_ratio)
        else:
            print('augmentation error')
            assert False

    if coordinates is not None:
        return data, coordinates
    else:
        return data


def drop_nodes_ori(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()

    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                  (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
    except:
        data = data

    return data


def drop_nodes(data, aug_ratio, coordinates=None):
    node_num, c, h, w = data.x.size()
    _, edge_num = data.edge_index.shape
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

    # edge_index = data.edge_index.numpy()
    edge_index = data.edge_index

    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                  (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
    except:
        data = data

    if coordinates is not None:
        coordinates = coordinates[idx_nondrop]
        return data, coordinates
    else:
        return data


def permute_edges_ori(data, aug_ratio):
    """
    only change edge_index, all the other keys unchanged and consistent
    """
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)
    edge_index = data.edge_index.numpy()

    idx_delete = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
    data.edge_index = data.edge_index[:, idx_delete]

    return data


def permute_edges(data, aug_ratio):
    """
    only change edge_index, all the other keys unchanged and consistent
    """
    node_num, c, h, w = data.x.size()
    _, edge_num = data.edge_index.shape
    permute_num = int(edge_num * aug_ratio)
    # edge_index = data.edge_index.numpy()

    idx_delete = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
    data.edge_index = data.edge_index[:, idx_delete]

    return data


def mask_nodes_ori(data, aug_ratio):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = token.clone().detach()

    return data


def mask_nodes(data, aug_ratio):
    node_num, c, h, w = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = token.clone().detach()

    return data