# 使用说明

- S集群：yanfang的newtorch或hest(目前存在问题)或allslide环境(Gigapath必须用newtorch或hest环境)，run和debug需要在code-server终端先proxy_off；第二种方式：通过pdb调试
- 3090：基于n1环境，环境问题需要确保以下包都已经安装，后面链接需要写torch版本和cuda版本：特别是需要安装pyg_lib

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
``` 

- torch-geometric文档链接：https://pytorch-geometric.readthedocs.io/en/latest/index.html

## 预训练：

- 本地运行命令(Prostate，Light网络, obsoleted)：

```bash
python pretrain.py --model GraphCL --gnn GCN --mode soft --dataset Prostate --encoder Pathoduet --encoder_path /mnt/data/smart_health_02/lujiaxuan/workingplace/GleasonGrade/code/PathoDuet/models/checkpoint_p2.pth --learning_rate 0.0001 --cluster_sizes 200 100 50 --batch_size 3 --num_parts 200 --num_workers 0 --checkpoint_suffix GCN_soft_pool_cluster_200_100_50_SGD_lr_0.0001_batch_3_worker_32
```

- 参数：
  - --mode
    - original：无池化的普通GCN
    - hard：forward函数内调用kmeans，内存开销极大，目前训练多个epoch后会爆内存
    - soft(优先)：模仿DiffPool，使用一个GNN来学习聚类的id

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

- 运行命令(TCGA_crop_FFPE): 
基于GigaPath-Tile作为patch encoder（Light网络）：
```bash
proxy_off
conda activate newtorch
cd /mnt/hwfile/smart_health/lujiaxuan/WSIGraph
python pretrain.py --model GraphCL --gnn GCN --mode soft --dataset TCGA --encoder GigaPath --encoder_path /mnt/hwfile/smart_health/lujiaxuan/hest/fm_v1/gigapath/pytorch_model.bin --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 8 --num_parts 500 --num_workers 32 --checkpoint_suffix GigaPath_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32
```

基于PathoDuet作为patch encoder（Light网络）：
```bash
proxy_off
conda activate allslide
cd /mnt/hwfile/smart_health/lujiaxuan/WSIGraph
python pretrain.py --model GraphCL --gnn GCN --mode soft --dataset TCGA --encoder Pathoduet --encoder_path /mnt/hwfile/smart_health/lujiaxuan/PathoDuet/models/checkpoint_p2.pth --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 8 --num_parts 500 --num_workers 32 --checkpoint_suffix GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32
```


- 初始化图可视化：参考visualize_graph()


## 下游微调

更多不同配置见scripts/finetune_JinYu.sh

- 运行命令(JinYu):
```bash
conda activate allslide
cd /mnt/hwfile/smart_health/lujiaxuan/WSIGraph
python downstream_tune.py --model GraphCL --gnn GCN --mode soft --combine_mode hier_weighted_mean --post_mode abmil --dataset JinYu --encoder Pathoduet --encoder_path /mnt/hwfile/smart_health/lujiaxuan/PathoDuet/models/checkpoint_p2.pth --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 8 --num_parts 500 --num_workers 32 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/TCGA.GraphCL.GCN.GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_27_loss_2.1094.pth --checkpoint_suffix FT_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32
```

以上配置参数量3,645,244

- 参数：
  - --combine_mode
    - linear_probing：进行linear probing，拿到1000多个768的token后取平均得到单独的1个768的token再过线性层
    - graph_level：直接返回图级别的特征
    - region_level：将区域特征进行平均
    - hier_mean(优先)：将每一级特征进行平均，再平均
    - hier_weighted_mean(待验证)：将每一级特征进行可学习权重的加权平均，再平均
  - --post_mode
    - linear_probing：冻结GNN，仅训练最终特征融合中的可学习权重和最终的分类头
 - --loss
   - WeightedCrossEntropyLoss: 加权的单标签分类
   - MultiLabelBCELoss：多标签分类


## 其它代码
- calculate_ceph_files.py：统计ceph上文件数量