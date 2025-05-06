#!/bin/bash
conda activate hest
cd /mnt/hwfile/smart_health/lujiaxuan/WSIGraph

pth_files=(
    "/mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/TCGA.GraphCL.GCN.GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_27_loss_2.1094.pth"
)

for pth_file in "${pth_files[@]}"
do
    filename=$(basename -- "$pth_file")
    filename="${filename%.*}"

    nohup python downstream_tune.py --model GraphCL --gnn GCN --mode soft --combine_mode hier_weighted_mean --dataset JinYu --encoder Pathoduet --encoder_path /mnt/hwfile/smart_health/lujiaxuan/PathoDuet/models/checkpoint_p2.pth --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 4 --num_parts 500 --num_workers 32 --loss WeightedCrossEntropyLoss --gnn_ckpt "$pth_file" --checkpoint_suffix GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32 > "FT_JinYu_${filename}.txt" 2>&1 &
done

# 最好的用--mode soft --combine_mode hier_weighted_mean --post_mode linear_probing

# patch encoder: Pathoduet, slide encoder:Ours
# freeze gcn, tuning cls head with linear probing
# nohup python downstream_tune.py --model GraphCL --gnn GCN --mode soft --combine_mode hier_weighted_mean --post_mode linear_probing --dataset JinYu --encoder Pathoduet --encoder_path /mnt/hwfile/smart_health/lujiaxuan/PathoDuet/models/checkpoint_p2.pth --learning_rate 0.001 --cluster_sizes 200 200 100 100 50 --batch_size 4 --num_parts 500 --num_workers 32 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/TCGA.GraphCL.GCN.GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_27_loss_2.1094.pth --checkpoint_suffix FT_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.001_batch_8_worker_32 > post-LP-4cards_lr0.001-batch4_JinYu_TCGA.GraphCL.GCN.GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.001_batch_8_worker_32_epoch_27_loss_2.1094.txt 2>&1 &

# patch encoder: GigaPath-Tile, slide encoder: GigaPath-Slide
# freeze gcn(GigaPath-Slide), tuning cls head with linear probing
nohup python downstream_tune.py --model GraphCL --gnn GCN --mode GigaPath --combine_mode hier_mean --post_mode linear_probing --dataset JinYu --encoder GigaPath --encoder_path /mnt/hwfile/smart_health/lujiaxuan/hest/fm_v1/gigapath/pytorch_model.bin --learning_rate 0.001 --cluster_sizes 200 200 100 100 50 --batch_size 2 --num_parts 500 --num_workers 0 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/prov-gigapath/slide_encoder.pth --checkpoint_suffix LP_GigaPath_LongNet_SGD_lr_0.001_batch_8_worker_32 > post-LP_GigaPath_LongNet-4cards_lr0.001-batch4_JinYu.txt 2>&1 &

# patch encoder: GigaPath-Tile, slide encoder: GigaPath-Slide
# freeze gcn(GigaPath-Slide), tuning cls head with abmil
nohup python downstream_tune.py --model GraphCL --gnn GCN --mode GigaPath --combine_mode abmil --post_mode abmil --dataset JinYu --encoder GigaPath --encoder_path /mnt/hwfile/smart_health/lujiaxuan/hest/fm_v1/gigapath/pytorch_model.bin --learning_rate 0.001 --cluster_sizes 200 200 100 100 50 --batch_size 2 --num_parts 500 --num_workers 0 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/prov-gigapath/slide_encoder.pth --checkpoint_suffix ABMIL_GigaPath_LongNet_SGD_lr_0.001_batch_8_worker_32 > post-ABMIL_GigaPath_LongNet-4cards_lr0.001-batch4_JinYu.txt 2>&1 &

# patch encoder: GigaPath-Tile, slide encoder: Ours
# freeze gcn(GigaPath-Slide), tuning cls head with abmil
(依然会segmentation fault，tmux5,158机器hest环境无错误，batch可大幅增大，需要测试精度？)nohup python downstream_tune.py --model GraphCL --gnn GCN --mode soft --combine_mode abmil --post_mode abmil --dataset JinYu --encoder GigaPath --encoder_path /mnt/hwfile/smart_health/lujiaxuan/hest/fm_v1/gigapath/pytorch_model.bin --learning_rate 0.001 --cluster_sizes 200 200 100 100 50 --batch_size 4 --num_parts 500 --num_workers 0 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/TCGA.GraphCL.GCN.GigaPath_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_97_loss_1.1986.pth --checkpoint_suffix ABMIL_GigaPath_TCGA.GraphCL.GCN.GigaPath_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_97_loss_1.1986 > post-ABMIL_GigaPath_GCN_soft_pool_cluster_200_200_100_100_50-4cards_lr0.001-batch4_JinYu.txt 2>&1 &


# patch encoder: Pathoduet, slide encoder:Ours
# tuning gcn and cls head
# nohup python downstream_tune.py --model GraphCL --gnn GCN --mode soft --combine_mode hier_weighted_mean --dataset JinYu --encoder Pathoduet --encoder_path /mnt/hwfile/smart_health/lujiaxuan/PathoDuet/models/checkpoint_p2.pth --learning_rate 0.001 --cluster_sizes 200 200 100 100 50 --batch_size 16 --num_parts 500 --num_workers 32 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/TCGA.GraphCL.GCN.GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_27_loss_2.1094.pth --checkpoint_suffix FT_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.001_batch_8_worker_32 > FT-8cards_lr0.001-batch16_JinYu_TCGA.GraphCL.GCN.GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.001_batch_8_worker_32_epoch_27_loss_2.1094.txt 2>&1 &


# patch encoder: Pathoduet, slide encoder:Ours
# linear probing
# nohup python downstream_tune.py --model GraphCL --gnn GCN --mode soft --combine_mode linear_probing --dataset JinYu --encoder Pathoduet --encoder_path /mnt/hwfile/smart_health/lujiaxuan/PathoDuet/models/checkpoint_p2.pth --learning_rate 0.001 --cluster_sizes 200 200 100 100 50 --batch_size 4 --num_parts 500 --num_workers 16 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/TCGA.GraphCL.GCN.GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_27_loss_2.1094.pth --checkpoint_suffix FT_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.001_batch_8_worker_32 > LP-4cards_lr0.001-batch4_JinYu_TCGA.GraphCL.GCN.GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.001_batch_8_worker_32_epoch_27_loss_2.1094.txt 2>&1 &