import glob
import os
import re
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
from scipy.spatial import KDTree
from torch_geometric.data import Data

import cv2
# import pdb

import torch
import torch.utils.data as data

from ProG.graph import graph_views, visualize_graph
from pathlib import Path

from petrel_client.client import Client

NR_CLASSES = 4


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


class TCGA_e2e(data.Dataset):
    """ TCGA dataset for E2E training, return the patch imgs rather than features

    Args:
        dataset_cfg (dict): Define from the config file(yaml).
        phase (str): 'train' or 'test'. If 'train', return the traindataset. If 'test', return the testdataset.

    """

    def __init__(self, dataset_cfg=None,
                 phase=None, prefix='TCGA_crop_FFPE'):
        
        # client = Client('/mnt/hwfile/smart_health/lujiaxuan/petreloss.conf')
        self.client = Client('/mnt/petrelfs/yanfang/.petreloss.conf')

        bucket_name = 'yanfang3'


        # url = 'yanfang:s3://yanfang3/TCGA_crop_FFPE/'
        url = f'yanfang:s3://{bucket_name}/{prefix}/'
        wsi_paths = self.client.list(url)

        # img_urls = [f'yanfang:s3://{bucket_name}/{wsi_path}' for wsi_path in wsi_paths]
        tcgas =[d for d in wsi_paths if 'TCGA' in d]

        self.data_urls = []
        for tcga in tcgas:
            url_prefix = url + tcga
            svs_urls = [url_prefix + d for d in self.client.list(url_prefix)]
            self.data_urls += svs_urls

            # random.shuffle(patch_urls)
            # self.data_urls += patch_urls[:int(selection * len(patch_urls))]
        # self.transform = transform
        random.shuffle(self.data_urls)

        if phase in ['train', 'finetune']:
            self.patch_transform = transforms.Compose(augmentation_train)
        elif phase in ['valid', 'test']:
            self.patch_transform = transforms.Compose(augmentation_test)
        self.phase = phase

    def __len__(self):
        return len(self.data_urls)

    def __getitem__(self, idx):
        data_dict = dict()  # {slide, slide_label, instance_label}
        slide_id = self.data_urls[idx].rstrip('/').split('/')[-1]
        data_dict.update(slide_id=slide_id)

        # slide_label = self.label[idx]
        slide_label = 0
        data_dict.update(slide_label=slide_label)

        # Load patch data from patch_512 and patch_256 folders
        patch_512_folder = os.path.join(self.data_urls[idx], 'patch_512')
        patch_256_folder = os.path.join(self.data_urls[idx], 'patch_256')

        imgs_512 = []
        imgs_256 = []

        # Use client.list to get file names from Ceph
        img_names_512 = self.client.list(patch_512_folder)
        img_names_256 = self.client.list(patch_256_folder)

        coordinates_512 = []
        coordinates_256 = []
        img_names_512 = list(img_names_512)
        img_names_256 = list(img_names_256)

        most_patch_num = 500
        if len(img_names_512) > most_patch_num: # only retain 1500 patches
            img_names_512 = random.sample(img_names_512, most_patch_num)
        if len(img_names_256) > most_patch_num: # only retain 1500 patches
            img_names_256 = random.sample(img_names_256, most_patch_num)
        least_patch_num = 200
        if len(img_names_512) < least_patch_num:
            img_names_512 += random.choices(img_names_512, k=least_patch_num - len(img_names_512))
        if len(img_names_256) < least_patch_num:
            img_names_256 += random.choices(img_names_256, k=least_patch_num - len(img_names_256))
        
        for img_name in img_names_512:
            img_path = os.path.join(patch_512_folder, img_name)
            img_bytes = self.client.get(img_path)
            if img_bytes is None:
                continue
            # assert (img_bytes is not None)
            img_mem_view = memoryview(img_bytes)
            img_array = np.frombuffer(img_mem_view, np.uint8)
            try:
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = transforms.ToPILImage()(img).convert('RGB')
            except:
                img = Image.new('RGB', (224, 224))
                print(img_path)
            if self.patch_transform is not None:
                img = self.patch_transform(img)
            imgs_512.append(img)
            coord = extract_coordinates(img_name)
            coordinates_512.append(coord)

        for img_name in img_names_256:
            img_path = os.path.join(patch_256_folder, img_name)
            img_bytes = self.client.get(img_path)
            if img_bytes is None:
                continue
            # assert (img_bytes is not None)
            img_mem_view = memoryview(img_bytes)
            img_array = np.frombuffer(img_mem_view, np.uint8)
            try:
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = transforms.ToPILImage()(img).convert('RGB')
            except:
                img = Image.new('RGB', (224, 224))
                print(img_path)
            if self.patch_transform is not None:
                img = self.patch_transform(img)
            imgs_256.append(img)
            coord = extract_coordinates(img_name)
            coordinates_256.append(coord)

        imgs_512 = torch.stack(imgs_512)
        imgs_256 = torch.stack(imgs_256) 
        data_dict.update(instance_img=imgs_512)

        # construct the graph
        coordinates_512 = np.array(coordinates_512)
        # coordinates_512 = normalize_coordinates(coordinates_512)
        distance_matrix = np.sqrt(((coordinates_512[:, np.newaxis] - coordinates_512)**2).sum(axis=2))
        distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        threshold_distances = np.sort(distances)[:int(len(distances) * 0.1)] # 0.1 can be changed, or random
        min_distance = np.mean(threshold_distances)
        edge_index = []
        tree = KDTree(coordinates_512)
        for i in range(len(coordinates_512)):
            # query_ball_point返回的是距离当前点distance_threshold内的所有点的索引
            indices = tree.query_ball_point(coordinates_512[i], min_distance)
            for j in indices:
                if i < j:  # 避免重复边和自环
                    edge_index.append([i, j])
        edge_index = np.array(edge_index).T
        data_dict.update(edge_index=edge_index)
        edge_index_512 = edge_index

        coordinates_256 = np.array(coordinates_256)
        # coordinates_256 = normalize_coordinates(coordinates_256)
        distance_matrix = np.sqrt(((coordinates_256[:, np.newaxis] - coordinates_256)**2).sum(axis=2))
        distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        threshold_distances = np.sort(distances)[:int(len(distances) * 0.1)] # 0.1 can be changed, or random
        min_distance = np.mean(threshold_distances)
        edge_index = []
        tree = KDTree(coordinates_256)
        for i in range(len(coordinates_256)):
            # query_ball_point返回的是距离当前点distance_threshold内的所有点的索引
            indices = tree.query_ball_point(coordinates_256[i], min_distance)
            for j in indices:
                if i < j:  # 避免重复边和自环
                    edge_index.append([i, j])
        edge_index = np.array(edge_index).T
        data_dict.update(edge_index=edge_index)
        edge_index_256 = edge_index

        # visualize the graph
        # visualize_graph(coordinates, edge_index, "graph.svg")

        g_1 = Data(x=imgs_512, edge_index=edge_index_512)
        g_2 = Data(x=imgs_256, edge_index=edge_index_256)

        if self.phase == 'train':
            # random augmentation
            aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
            aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
            aug_ratio = random.randint(1, 3) * 1.0 / 10  # 0.1,0.2,0.3

            view_1, coord_1 = graph_views(data=g_1, aug=aug1, aug_ratio=aug_ratio, coordinates=coordinates_512)
            view_1 = Data(x=view_1.x, edge_index=view_1.edge_index)
            view_2, coord_2 = graph_views(data=g_2, aug=aug2, aug_ratio=aug_ratio, coordinates=coordinates_256)
            view_2 = Data(x=view_2.x, edge_index=view_2.edge_index)
        else:
            view_1, view_2 = g_1, g_2
            coord_1, coord_2 = coordinates_512, coordinates_256

        view_1 = np.array([view_1, coord_1])
        view_2 = np.array([view_2, coord_2])

        # pdb.set_trace()

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
            # 'base_path': '/mnt/data/smart_health_02/lujiaxuan/workingplace/GleasonGrade/segResData1/features-pathoduet-p2/pt_files',
            # loading the features data
            # 'slide_path': '/mnt/data/oss_beijing/transfers/lujiaxuan/GleasonGrade/rawdata',
            # 'fold': 0,
            # 'nfolds': 4,
            'max_patch_num': 500,  # 200, For deployment adjust according to hardware capability
            # 'label_dir': '/mnt/data/smart_health_02/transfers/lujiaxuan/workingplace/GleasonGrade/code/Mixed_supervision/dataset_csv/prostate/',
            'data_load_mode': 0,  # 0: all data info in a single .bin file, 1: use SegGini loading method
            'train_dataset': {
                'image_dir': '',  # Assuming image_dir is to be set or is empty as per given structure
                'batch_size': 16,  # Adjust based on available resources
                # 'num_workers': 4,  # Adjust based on local debug or cloud service submission
            },
            'test_dataset': {
                'batch_size': 1,  # Adjust based on available resources
                # 'num_workers': 32,  # Adjust based on local debug or cloud service submission
            }
        }


def TCGA(num_parts=200, phase='train'):
    cfg = Config()
    Mydata = TCGA_e2e(cfg.Data, phase=phase)

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

    TCGA()

