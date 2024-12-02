import os
import numpy as np
import random
import torch
import torch.nn as nn
from copy import deepcopy
from random import shuffle
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph, k_hop_subgraph
from torch.nn.utils.rnn import pad_sequence

import torch.nn.functional as F

from ProG.dataset.prostate_RJ import Prostate
from ProG.dataset.CiteSeer import CiteSeer
from ProG.dataset.TCGA import TCGA
from ProG.dataset.TCGA_frozen import TCGA_frozen
from ProG.dataset.JinYu import JinYu
from ProG.dataset.Combined import Combined

seed = 0

datasets = {'CiteSeer': CiteSeer, 'Prostate': Prostate, 'TCGA': TCGA, 'TCGA_frozen': TCGA_frozen,
        'JinYu': JinYu, 'Combined': Combined}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))
    else:
        print("folder exists! {}".format(path))


# used in pre_train.py
def gen_ran_output(data, model):
    vice_model = deepcopy(model)

    for (vice_name, vice_model_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if vice_name.split('.')[0] == 'projection_head':
            vice_model_param.data = param.data
        else:
            vice_model_param.data = param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std())
    z2 = vice_model.forward_cl(data.x, data.edge_index, data.batch)

    return z2


# used in pre_train.py
def load_data4pretrain(dataname='CiteSeer', num_parts=200, phase='train', encoder=None):
    graph_list = datasets[dataname](num_parts=num_parts, phase=phase, encoder=encoder)

    return graph_list


def ensure_tensor(data, dtype):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).to(dtype=dtype)
    elif isinstance(data, torch.Tensor):
        tensor = data.to(dtype=dtype)

    return tensor

# def collate_fn(data_list):
    view_1_list, view_2_list = [], []
    coord_1_list, coord_2_list = [], []
    file_path_list = []
    # for view_1, view_2 in data_list:
    #     view_1.edge_index = torch.tensor(view_1.edge_index, dtype=torch.long)
    #     view_2.edge_index = torch.tensor(view_2.edge_index, dtype=torch.long)
    #     view_1_list.append(view_1)
    #     view_2_list.append(view_2)
    for item in data_list:
        view_1, coord_1 = item[0][0], item[0][1]
        view_2, coord_2 = item[1][0], item[1][1]

        # view_1.edge_index = torch.tensor(view_1.edge_index, dtype=torch.long)
        # view_2.edge_index = torch.tensor(view_2.edge_index, dtype=torch.long)
        # view_1_list.append(view_1)
        # view_2_list.append(view_2)
        # coord_1_list.append(torch.tensor(coord_1, dtype=torch.float))
        # coord_2_list.append(torch.tensor(coord_2, dtype=torch.float))
        view_1.edge_index = ensure_tensor(view_1.edge_index, dtype=torch.long)
        view_2.edge_index = ensure_tensor(view_2.edge_index, dtype=torch.long)
        view_1_list.append(view_1)
        view_2_list.append(view_2)
        coord_1_list.append(ensure_tensor(coord_1, dtype=torch.float))
        coord_2_list.append(ensure_tensor(coord_2, dtype=torch.float))
        file_path_list.append(item[2])

    batch_view_1 = Batch.from_data_list(view_1_list)
    batch_view_2 = Batch.from_data_list(view_2_list)
    # batch_coord_1 = pad_sequence(coord_1_list, batch_first=True, padding_value=0)
    # batch_coord_2 = pad_sequence(coord_2_list, batch_first=True, padding_value=0)
    batch_coord_1 = torch.cat(coord_1_list, dim=0)
    batch_coord_2 = torch.cat(coord_2_list, dim=0)

    if len(data_list[0]) == 3: # have label
        label_list = []
        for item in data_list:
            label_list.append(item[2])
        return batch_view_1, batch_view_2, batch_coord_1, batch_coord_2, torch.tensor(label_list), file_path_list
    else:
        return batch_view_1, batch_view_2, batch_coord_1, batch_coord_2, file_path_list

def collate_fn(data_list):
    view_1_list, view_2_list = [], []
    coord_1_list, coord_2_list = [], []
    label_list = []
    file_path_list = []
    has_label = False  # 标记是否存在标签

    for item in data_list:
        # 解包视图和坐标
        view_1, coord_1 = item[0][0], item[0][1]
        view_2, coord_2 = item[1][0], item[1][1]

        # 确保 edge_index 是张量
        view_1.edge_index = ensure_tensor(view_1.edge_index, dtype=torch.long)
        view_2.edge_index = ensure_tensor(view_2.edge_index, dtype=torch.long)
        
        # 添加到列表
        view_1_list.append(view_1)
        view_2_list.append(view_2)
        coord_1_list.append(ensure_tensor(coord_1, dtype=torch.float))
        coord_2_list.append(ensure_tensor(coord_2, dtype=torch.float))

        if len(item) == 4:
            # 数据项包含标签和文件路径
            has_label = True
            label = item[2]
            file_path = item[3]
            label_list.append(label)
        elif len(item) == 3:
            # 数据项仅包含文件路径
            file_path = item[2]
        else:
            raise ValueError("数据项的长度必须是3（无标签）或4（有标签）")
        
        file_path_list.append(file_path)

    # 创建批次
    batch_view_1 = Batch.from_data_list(view_1_list)
    batch_view_2 = Batch.from_data_list(view_2_list)
    batch_coord_1 = torch.cat(coord_1_list, dim=0)
    batch_coord_2 = torch.cat(coord_2_list, dim=0)

    if has_label:
        labels = torch.tensor(label_list, dtype=torch.long)  # 根据需要调整 dtype
        return (batch_view_1, batch_view_2, batch_coord_1, batch_coord_2, labels), file_path_list
    else:
        return (batch_view_1, batch_view_2, batch_coord_1, batch_coord_2), file_path_list

def get_model(model):
    # check whether the model is wrapped by DataParallel
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    else:
        return model


# used in prompt.py
def act(x=None, act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        if x is None:
            return torch.nn.LeakyReLU()
        else:
            return F.leaky_relu(x)
    elif act_type == 'tanh':
        if x is None:
            return torch.nn.Tanh()
        else:
            return torch.tanh(x)


def print_model_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
        print(f"{name} has {param} parameters")
    print(f"Total Parameters: {total_params}")


class Gprompt_tuning_loss(nn.Module):
    def __init__(self, tau=0.1):
        super(Gprompt_tuning_loss, self).__init__()
        self.tau = tau

    def forward(self, embedding, center_embedding, labels):
        # 对于每个样本对（xi,yi), loss为 -ln(sim正 / sim正+sim负)

        # 计算所有样本与所有个类原型的相似度
        similarity_matrix = F.cosine_similarity(embedding.unsqueeze(1), center_embedding.unsqueeze(0),
                                                dim=-1) / self.tau
        exp_similarities = torch.exp(similarity_matrix)
        # Sum exponentiated similarities for the denominator
        pos_neg = torch.sum(exp_similarities, dim=1, keepdim=True)
        # select the exponentiated similarities for the correct classes for the every pair (xi,yi)
        pos = exp_similarities.gather(1, labels.view(-1, 1))
        L_prompt = -torch.log(pos / pos_neg)
        loss = torch.sum(L_prompt)

        return loss


def distance2center(input,center):
    n = input.size(0)
    k = center.size(0)
    input_power = torch.sum(input * input, dim=1, keepdim=True).expand(n, k)
    center_power = torch.sum(center * center, dim=1).expand(n, k)

    distance = input_power + center_power - 2 * torch.mm(input, center.transpose(0, 1))
    return distance


def center_embedding(input, index, label_num):
    device=input.device
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index.unsqueeze(1).expand(-1, input.size(1)), src=input)
    class_counts = torch.bincount(index, minlength=label_num).unsqueeze(1).to(dtype=input.dtype, device=device)

    # Take the average embeddings for each class
    # If directly divided the variable 'c', maybe encountering zero values in 'class_counts', such as the class_counts=[[0.],[4.]]
    # So we need to judge every value in 'class_counts' one by one, and seperately divided them.
    # output_c = c/class_counts
    for i in range(label_num):
        if(class_counts[i].item()==0):
            continue
        else:
            c[i] /= class_counts[i]

    return c, class_counts


def constraint(device,prompt):
    if isinstance(prompt,list):
        sum=0
        for p in prompt:
            sum=sum+torch.norm(torch.mm(p,p.T)-torch.eye(p.shape[0]).to(device))
        return sum/len(prompt)
    else:
        return torch.norm(torch.mm(prompt,prompt.T)-torch.eye(prompt.shape[0]).to(device))


def __seeds_list__(nodes):
    split_size = max(5, int(nodes.shape[0] / 400))
    seeds_list = list(torch.split(nodes, split_size))
    if len(seeds_list) < 400:
        print('len(seeds_list): {} <400, start overlapped split'.format(len(seeds_list)))
        seeds_list = []
        while len(seeds_list) < 400:
            split_size = random.randint(3, 5)
            seeds_list_1 = torch.split(nodes, split_size)
            seeds_list = seeds_list + list(seeds_list_1)
            nodes = nodes[torch.randperm(nodes.shape[0])]
    shuffle(seeds_list)
    seeds_list = seeds_list[0:400]

    return seeds_list


def __dname__(p, task_id):
    if p == 0:
        dname = 'task{}.meta.train.support'.format(task_id)
    elif p == 1:
        dname = 'task{}.meta.train.query'.format(task_id)
    elif p == 2:
        dname = 'task{}.meta.test.support'.format(task_id)
    elif p == 3:
        dname = 'task{}.meta.test.query'.format(task_id)
    else:
        raise KeyError

    return dname


def __pos_neg_nodes__(labeled_nodes, node_labels, i: int):
    pos_nodes = labeled_nodes[node_labels[:, i] == 1]
    pos_nodes = pos_nodes[torch.randperm(pos_nodes.shape[0])]
    neg_nodes = labeled_nodes[node_labels[:, i] == 0]
    neg_nodes = neg_nodes[torch.randperm(neg_nodes.shape[0])]
    return pos_nodes, neg_nodes


def __induced_graph_list_for_graphs__(seeds_list, label, p, num_nodes, potential_nodes, ori_x, same_label_edge_index,
                                      smallest_size, largest_size):
    seeds_part_list = seeds_list[p * 100:(p + 1) * 100]
    induced_graph_list = []
    for seeds in seeds_part_list:

        subset, _, _, _ = k_hop_subgraph(node_idx=torch.flatten(seeds), num_hops=1, num_nodes=num_nodes,
                                         edge_index=same_label_edge_index, relabel_nodes=True)

        temp_hop = 1
        while len(subset) < smallest_size and temp_hop < 5:
            temp_hop = temp_hop + 1
            subset, _, _, _ = k_hop_subgraph(node_idx=torch.flatten(seeds), num_hops=temp_hop, num_nodes=num_nodes,
                                             edge_index=same_label_edge_index, relabel_nodes=True)

        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            candidate_nodes = torch.from_numpy(np.setdiff1d(potential_nodes.numpy(), subset.numpy()))

            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]

            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            # directly downmsample
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - len(seeds)]
            subset = torch.unique(torch.cat([torch.flatten(seeds), subset]))

        sub_edge_index, _ = subgraph(subset, same_label_edge_index, num_nodes=num_nodes, relabel_nodes=True)

        x = ori_x[subset]
        graph = Data(x=x, edge_index=sub_edge_index, y=label)
        induced_graph_list.append(graph)

    return induced_graph_list


