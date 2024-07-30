import glob
import os
import re
import json
import sys
sys.path.append('.')
from tqdm import tqdm
# from utils.util import read_yaml
import torchvision.transforms as transforms
import random
import numpy as np
import torch
import pandas as pd
from PIL import Image, ImageFilter
import h5py
import openslide
try:
    from pathadox_allslide import AllSlide
except Exception:
    def add_to_sys_path(dir_path):
        sys.path.append(dir_path)
        for root, dirs, files in os.walk(dir_path):
            sys.path.append(root)
    add_to_sys_path("/mnt/hwfile/smart_health/lujiaxuan/allslide")
    import AllSlide
from scipy.spatial import KDTree
from torch_geometric.data import Data

import cv2
# import pdb

import torch
import torch.utils.data as data

from ProG.graph import graph_views, visualize_graph
from pathlib import Path

from petrel_client.client import Client

NR_CLASSES = 21


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# the augmentation in Pathoduet-p2
augmentation_train = [
    transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
]

augmentation_test = [
    transforms.CenterCrop(224),
    transforms.ToTensor()
]




def sort_key_function(path):
    """sort the path according to the number part of the filename"""
    basename = os.path.basename(path)
    number_part = int(basename.split('_')[-1].split('.')[0])
    return number_part


def to_mapper(df):
    """ map the raw label into vector

    Args:
        df (DataFrame): Record the dataframe of the data label
    Returns:
        dict:
            E.g.: {'xxx': array([0,0,1,1]),
                    ...
                    }
    """
    mapper = dict()
    for name, row in df.iterrows():
        mapper[name] = np.array(
            [row["benign"], row["grade3"], row["grade4"], row["grade5"]]
        )
    return mapper


def extract_coordinates(string):
    match = re.search(r'_(\d+)_(\d+)_X\d+_\d+\.jpg$', string)
    if match:
        # 提取匹配到的坐标并转换为整数
        x, y = map(int, match.groups())
        return (x, y)
    else:
        return (100, 100)

def normalize_coordinates(coordinates):
    x_min, y_min = coordinates.min(axis=0)
    x_max, y_max = coordinates.max(axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    normalized_array = (coordinates - [x_min, y_min]) / [x_range, y_range]
    return normalized_array


class JinYu_e2e(data.Dataset):
    """ JinYu private dataset for finetuning

    Args:
        dataset_cfg (dict): Define from the config file(yaml).
        phase (str): 'train' or 'test'. If 'train', return the traindataset. If 'test', return the testdataset.

    """

    def __init__(self, dataset_cfg=None,
                 phase=None, prefix='JinYu'):
        
        self.dataset_cfg = dataset_cfg
        self.fold = self.dataset_cfg['fold']
        self.base_path = self.dataset_cfg['base_path']
        self.h5_path = self.dataset_cfg['h5_path']
        split_file_path = self.dataset_cfg['label_dir'] + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(split_file_path)

        self.slide_id_2_paths = dict()
        self.slide_path = self.dataset_cfg['slide_path']
        self.slide_paths = glob.glob(os.path.join(self.base_path, '*.h5')) # for patch data
        # self.slide_paths = glob.glob(os.path.join(self.base_path, '*.pt')) # for features data
        for path in self.slide_paths:
            slide_id = Path(path).stem
            self.slide_id_2_paths[slide_id] = path

        if phase in ['train', 'finetune']:
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()

        if phase == 'valid':
            self.data = self.slide_data.loc[:, 'val'].dropna()
            self.label = self.slide_data.loc[:, 'val_label'].dropna()

        if phase == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

        combined = list(zip(self.data, self.label))
        random.shuffle(combined)
        self.data, self.label = zip(*combined)

        if phase in ['train', 'finetune']:
            self.patch_transform = transforms.Compose(augmentation_train)
        elif phase in ['valid', 'test']:
            self.patch_transform = transforms.Compose(augmentation_test)
        self.phase = phase

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = dict()  # {slide, slide_label, instance_label}
        # info = self.data_info[idx]
        # slide_id = info['svs_url'].rstrip('/').split('/')[-1]
        slide_id = self.data[idx]
        data_dict.update(slide_id=slide_id)

        slide_label = self.label[idx]
        data_dict.update(slide_label=slide_label)

        # load the patch data
        h5_path = self.slide_id_2_paths[slide_id]
        slide_id = slide_id.replace('&', '/').split('切片扫描/')[-1]
        slide_path = os.path.join(self.slide_path, slide_id + '.kfb')
        wsi = AllSlide.AllSlide(slide_path)
        imgs_1 = []
        imgs_2 = []
        coordinates = []
        with h5py.File(h5_path, 'r') as hdf5_file:
            coords = hdf5_file['coords']
            patch_level = coords.attrs['patch_level']
            patch_size = coords.attrs['patch_size']

            # # random choice when over 500 patches exists
            # if len(coords) > self.dataset_cfg['max_patch_num']:
            #     coords = random.sample(list(coords), self.dataset_cfg.max_patch_num)

            for coord in coords:
                # rescale the coord for kfb and sdpc
                if slide_path.split('.')[-1] in ['kfb', 'sdpc']:
                    coord[0] = int(coord[0] * wsi.level_dimensions[patch_level][0] / wsi.level_dimensions[0][0])
                    coord[1] = int(coord[1] * wsi.level_dimensions[patch_level][1] / wsi.level_dimensions[0][1])
                
                img = wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')
                if self.patch_transform is not None:
                    img_1 = self.patch_transform(img)  # in Pathoduet-p2
                    img_2 = self.patch_transform(img)  # in Pathoduet-p2
                    imgs_1.append(img_1)
                    imgs_2.append(img_2)
                else:
                    imgs_1.append(img)
                    imgs_2.append(img)
                coordinates.append(coord)

        imgs_1 = torch.stack(imgs_1)
        imgs_2 = torch.stack(imgs_2)
        data_dict.update(instance_img=imgs_1)

        # # load the features data
        # full_path = Path(self.base_path) / f'{slide_id}.pt'
        # features = torch.load(full_path)
        # data_dict.update(instance_feature=features.squeeze(1))

        # construct the graph
        coordinates = np.array(coordinates)
        distance_matrix = np.sqrt(((coordinates[:, np.newaxis] - coordinates)**2).sum(axis=2))
        distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        threshold_distances = np.sort(distances)[:int(len(distances) * 0.1)] # 0.1 can be changed, or random
        min_distance = np.mean(threshold_distances)
        edge_index = []
        tree = KDTree(coordinates)
        for i in range(len(coordinates)):
            # query_ball_point返回的是距离当前点distance_threshold内的所有点的索引
            indices = tree.query_ball_point(coordinates[i], min_distance)
            for j in indices:
                if i < j:  # 避免重复边和自环
                    edge_index.append([i, j])

        edge_index = np.array(edge_index).T
        data_dict.update(edge_index=edge_index)


        # visualize the graph
        # visualize_graph(coordinates, edge_index, "graph.svg")

        g_1 = Data(x=imgs_1, edge_index=edge_index)
        g_2 = Data(x=imgs_2, edge_index=edge_index)

        if self.phase == 'train':
            # random augmentation
            aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
            aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
            aug_ratio = random.randint(1, 3) * 1.0 / 10  # 0.1,0.2,0.3

            view_1, coord_1 = graph_views(data=g_1, aug=aug1, aug_ratio=aug_ratio, coordinates=coordinates)
            view_1 = Data(x=view_1.x, edge_index=view_1.edge_index)
            view_2, coord_2 = graph_views(data=g_2, aug=aug2, aug_ratio=aug_ratio, coordinates=coordinates)
            view_2 = Data(x=view_2.x, edge_index=view_2.edge_index)
        else:
            view_1, view_2 = g_1, g_2
            coord_1, coord_2 = coordinates, coordinates

        view_1 = np.array([view_1, coord_1])
        view_2 = np.array([view_2, coord_2])

        if self.phase == 'train':
            return view_1, view_2
        else:
            return view_1, view_2, slide_label


# def Train_dataset(cfg, phase='train'):
#     dataset = Prostate_e2e(cfg.Data, phase=phase)
#     return dataset
#
#
# def Test_dataset(cfg, phase='test'):
#     dataset = Prostate_e2e(cfg.Data, phase=phase)
#     return dataset


class Config:
    def __init__(self):
        self.Data = {
            # 'base_path': '/mnt/data/smart_health_02/transfers/lujiaxuan/workingplace/GleasonGrade/segResData1/patches', # loading the patch data
            'base_path': '/mnt/hwfile/smart_health/yanfang/evaluation/code/LymphAI/code/1-HEClassification/CLAM/patches/patches',
            # loading the features data
            'slide_path': '/mnt/hwfile/smart_health/yanfang/evaluation/dataset/切片扫描',
            'h5_path': '/mnt/petrelfs/yanfang/workspace/hw/yanfang/evaluation/code/LymphAI/code/1-HEClassification/CLAM/patches/patches',
            'fold': 0,
            'nfolds': 4,
            'max_patch_num': 500,  # 200, For deployment adjust according to hardware capability
            'label_dir': '/mnt/petrelfs/yanfang/workspace/hw/yanfang/evaluation/code/LymphAI/code/1-HEClassification/CLAM/splits/task_4_JINYU_subtyping_100_TransMIL/',
            'data_load_mode': 0,  # 0: all data info in a single .bin file, 1: use SegGini loading method
            'train_dataset': {
                'image_dir': '',  # Assuming image_dir is to be set or is empty as per given structure
                'batch_size': 16,  # Adjust based on available resources
                # 'num_workers': 4,  # Adjust based on local debug or cloud service submission
            },
            'test_dataset': {
                'batch_size': 2,  # Adjust based on available resources
                # 'num_workers': 32,  # Adjust based on local debug or cloud service submission
            }
        }


def JinYu(num_parts=200, phase='train'):
    cfg = Config()
    Mydata = JinYu_e2e(cfg.Data, phase=phase)

    # print(Mydata[0])
    # dataloader = DataLoader(Mydata)

    # for i, data in (enumerate(dataloader)):
    #     print(data)
    #     pass
    return Mydata

    # items = [item for item in list(dataloader)]
    # graph_list = [Data(x=item['instance_img'],edge_index=item['edge_index']) for item in items]
    #
    # # x = [item['instance_feature'][0] for item in list(dataloader)]
    #
    # input_dim = x[0].shape[-1]
    # hid_dim = input_dim



    # x = data.x.detach()
    # edge_index = data.edge_index
    # edge_index = to_undirected(edge_index)
    # data = Data(x=x, edge_index=edge_index)
    #
    #
    # graph_list = list(ClusterData(data=data, num_parts=num_parts, save_dir='Dataset/{}/'.format('CiteSeer')))
    # # graph_list = [data] # do not partition the graph

    # return graph_list, input_dim, hid_dim


if __name__ == '__main__':
    # # cfg = read_yaml('configs/prostate_e2e.yaml')
    # cfg = Config()
    #
    # # Mydata = Train_dataset(cfg=cfg, phase='train')
    # Mydata = Prostate_e2e(cfg.Data, phase='train')
    # # Mydata = Prostate_e2e(cfg.Data, phase='test')
    # dataloader = DataLoader(Mydata)
    # for i, data in (enumerate(dataloader)):
    #     pass

    JinYu()

