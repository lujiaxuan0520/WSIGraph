# 使用说明

- 基于n1环境，环境问题需要确保以下包都已经安装，后面链接需要写torch版本和cuda版本：特别是需要安装pyg_lib

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
``` 

- torch-geometric文档链接：https://pytorch-geometric.readthedocs.io/en/latest/index.html

## 预训练：

- 本地运行命令(Prostate)：

```bash
python pretrain.py --model GraphCL --gnn GCN --mode soft --dataset Prostate --encoder Pathoduet --encoder_path /mnt/data/smart_health_02/lujiaxuan/workingplace/GleasonGrade/code/PathoDuet/models/checkpoint_p2.pth --learning_rate 0.0001 --cluster_sizes 200 100 50 --batch_size 2 --num_parts 200 --num_workers 0 --checkpoint_suffix GCN_soft_pool_cluster_200_100_50_SGD_lr_0.0001_batch_2_worker_32
```

- 参数：
  - --mode
    - original：无池化的普通GCN
    - hard：forward函数内调用kmeans，内存开销极大，目前训练多个epoch后会爆内存
    - soft：模仿DiffPool，使用一个GNN来学习聚类的id

- 阿里云任务提交执行命令(Prostate)：单卡batch_size=3, num_workers=16/32；双卡batch_size=8, num_workers=32/64；4卡batch_size=16, num_workers=64/128

```bash
export PATH=/mnt/data/smart_health_02/lujiaxuan/anaconda3/envs/n1/bin:$PATH
sudo apt-get update -y
pip install timm~=0.6.13
sudo apt-get install openslide-tools -y
sudo apt-get install python-openslide -y
pip install openslide-python
cd /mnt/data/smart_health_02/lujiaxuan/workingplace/WSIGraph/code/ProG
python pretrain.py --model GraphCL --gnn GCN --mode soft --dataset Prostate --encoder Pathoduet --encoder_path /mnt/data/smart_health_02/lujiaxuan/workingplace/GleasonGrade/code/PathoDuet/models/checkpoint_p2.pth --learning_rate 0.0001 --cluster_sizes 200 100 50 --batch_size 3 --num_parts 200 --num_workers 32 --checkpoint_suffix GCN_soft_pool_cluster_200_100_50_SGD_lr_0.0001_batch_3_worker_32
```

- 初始化图可视化：参考visualize_graph()

## 下游微调

