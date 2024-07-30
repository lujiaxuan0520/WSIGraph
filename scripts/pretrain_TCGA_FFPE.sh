#!/bin/bash
proxy_off
conda activate hest
cd /mnt/hwfile/smart_health/lujiaxuan/WSIGraph

# patch encoder: use pathoduet
# nohup python pretrain.py --model GraphCL --gnn GCN --mode soft --dataset TCGA --encoder Pathoduet --encoder_path /mnt/hwfile/smart_health/lujiaxuan/PathoDuet/models/checkpoint_p2.pth --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 8 --num_parts 500 --num_workers 32 --resume_ckpt /mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/TCGA.GraphCL.GCN.GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_27_loss_2.1094.pth --checkpoint_suffix GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32 > GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32.txt 2>&1 &

# patch encoder: use GigaPath-Tile
nohup python pretrain.py --model GraphCL --gnn GCN --mode soft --dataset TCGA --encoder GigaPath --encoder_path /mnt/hwfile/smart_health/lujiaxuan/hest/fm_v1/gigapath/pytorch_model.bin --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 8 --num_parts 500 --num_workers 32 --resume_ckpt /mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/TCGA.GraphCL.GCN.GigaPath_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_3_loss_2.8624.pth --checkpoint_suffix GigaPath_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32 > GigaPath_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32.txt 2>&1 &