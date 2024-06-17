#!/bin/bash
conda activate allslide
cd /mnt/hwfile/smart_health/lujiaxuan/WSIGraph

pth_files=(
    "/mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/TCGA.GraphCL.GCN.GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_27_loss_2.1094.pth"
)

for pth_file in "${pth_files[@]}"
do
    filename=$(basename -- "$pth_file")
    filename="${filename%.*}"

    nohup python downstream_tune.py --model GraphCL --gnn GCN --mode soft --combine_mode hier_weighted_mean --dataset JinYu --encoder Pathoduet --encoder_path /mnt/hwfile/smart_health/lujiaxuan/PathoDuet/models/checkpoint_p2.pth --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 8 --num_parts 500 --num_workers 32 --gnn_ckpt "$pth_file" --checkpoint_suffix GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32 > "FT_JinYu_${filename}.txt" 2>&1 &
done

# nohup python downstream_tune.py --model GraphCL --gnn GCN --mode soft --combine_mode hier_weighted_mean --dataset JinYu --encoder Pathoduet --encoder_path /mnt/hwfile/smart_health/lujiaxuan/PathoDuet/models/checkpoint_p2.pth --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 8 --num_parts 500 --num_workers 32 --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/TCGA.GraphCL.GCN.GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_27_loss_2.1094.pth --checkpoint_suffix FT_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32 > FT_JinYu_TCGA.GraphCL.GCN.GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_27_loss_2.1094.txt 2>&1 &