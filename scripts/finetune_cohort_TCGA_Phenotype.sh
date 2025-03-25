#!/bin/bash
# 病理分期预测：
# ajcc_pathologic_stage.diagnoses：AJCC病理分期，例如 Stage III。
# ajcc_pathologic_t.diagnoses：AJCC病理T分期，例如 T3。
# ajcc_clinical_m.diagnoses：AJCC临床M分期，例如 M0 表示无远处转移。
# ajcc_pathologic_n.diagnoses：AJCC病理N分期，例如 N0 表示无淋巴结转移。

# 参数：如--dataset Cohort_TCGA_ACC_Phenotype_ajcc_stage
# prefix：
#     如Cohort_TCGA_ACC_Phenotype
# label：
#     ajcc_stage：AJCC总体病理分期，包括Stage I、Stage II、Stage III、Stage IV。
#     ajcc_t_stage：AJCC病理T分期，包括T1、T2、T3、T4。
#     ajcc_m_stage：AJCC临床M分期，包括M0、M1。
#     ajcc_n_stage：AJCC病理N分期，包括N1、N2。

# conda activate hest (针对GigaPath-Slide)
conda activate newtorch
cd /mnt/hwfile/smart_health/lujiaxuan/WSIGraph
proxy_off

pth_files=(
    "/mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/TCGA.GraphCL.GCN.GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_27_loss_2.1094.pth"
)

for pth_file in "${pth_files[@]}"
do
    filename=$(basename -- "$pth_file")
    filename="${filename%.*}"

    nohup python downstream_tune.py --model GraphCL --gnn GCN --mode soft --combine_mode hier_weighted_mean --dataset JinYu --encoder Pathoduet --encoder_path /mnt/hwfile/smart_health/lujiaxuan/PathoDuet/models/checkpoint_p2.pth --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 4 --num_parts 500 --num_workers 32 --loss WeightedCrossEntropyLoss --gnn_ckpt "$pth_file" --checkpoint_suffix GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32 > "FT_JinYu_${filename}.txt" 2>&1 &
done

# ---patch encoder: GigaPath-Tile---
# patch encoder: GigaPath-Tile, slide encoder: Ours
# freeze gcn(GigaPath-Slide), tuning cls head with abmil
nohup python downstream_tune.py --model GraphCL --gnn GCN --mode soft --combine_mode abmil --post_mode abmil --dataset Cohort_TCGA_ACC_Phenotype_ajcc_stage --encoder GigaPath --encoder_path /mnt/hwfile/smart_health/lujiaxuan/hest/fm_v1/gigapath/pytorch_model.bin --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 16 --num_parts 500 --num_workers 0 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/TCGA.GraphCL.GCN.GigaPath_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_97_loss_1.1986.pth --checkpoint_suffix ABMIL_GigaPath_TCGA.GraphCL.GCN.GigaPath_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_97_loss_1.1986 > post-ABMIL_GigaPath_GCN_soft_pool_cluster_200_200_100_100_50-4cards_lr0.001-batch4_JinYu.txt 2>&1 &

# patch encoder: GigaPath-Tile, slide encoder: GigaPath-Slide
# freeze gcn(GigaPath-Slide), tuning cls head with abmil
python downstream_tune.py --model GraphCL --gnn GCN --mode GigaPath --combine_mode abmil --post_mode abmil --dataset Cohort_TCGA_ACC_Phenotype_ajcc_stage --encoder GigaPath --encoder_path /mnt/hwfile/smart_health/lujiaxuan/hest/fm_v1/gigapath/pytorch_model.bin --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 16 --num_parts 500 --num_workers 0 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/prov-gigapath/slide_encoder.pth

# patch encoder: GigaPath-Tile, slide encoder: Random-GNN
# freeze gcn(GigaPath-Slide), tuning cls head with abmil
python downstream_tune.py --model GraphCL --gnn GCN --mode soft-random --combine_mode abmil --post_mode abmil --dataset Cohort_TCGA_ACC_Phenotype_ajcc_stage --encoder GigaPath --encoder_path /mnt/hwfile/smart_health/lujiaxuan/hest/fm_v1/gigapath/pytorch_model.bin --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 16 --num_parts 500 --num_workers 0 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/TCGA.GraphCL.GCN.GigaPath_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_97_loss_1.1986.pth --checkpoint_suffix ABMIL_GigaPath_TCGA.GraphCL.GCN.GigaPath_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_97_loss_1.1986


# ---patch encoder: UNI---
# patch encoder: UNI, slide encoder: Ours
# freeze gcn, tuning cls head with abmil
python downstream_tune.py --model GraphCL --gnn GCN --mode soft --combine_mode abmil --post_mode abmil --dataset Cohort_TCGA_ACC_Phenotype_ajcc_stage --encoder UNI --encoder_path /mnt/hwfile/smart_health/lujiaxuan/UNI/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 16 --num_parts 500 --num_workers 0 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/Combined.GraphCL.GCN.UNI_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_17_loss_2.2827.pth --checkpoint_suffix ABMIL_UNI_Combined.GraphCL.GCN.UNI_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_17_loss_2.2827

# patch encoder: UNI, slide encoder: Random-GNN
# freeze gcn, tuning cls head with abmil
python downstream_tune.py --model GraphCL --gnn GCN --mode soft-random --combine_mode abmil --post_mode abmil --dataset Cohort_TCGA_ACC_Phenotype_ajcc_stage --encoder UNI --encoder_path /mnt/hwfile/smart_health/lujiaxuan/UNI/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 16 --num_parts 500 --num_workers 0 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/Combined.GraphCL.GCN.UNI_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_4_loss_2.504.pth --checkpoint_suffix ABMIL_UNI_Combined.GraphCL.GCN.UNI_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_17_loss_2.2827


# ---patch encoder: PathOrchestra---
# patch encoder: PathOrchestra, slide encoder: Ours
# freeze gcn, tuning cls head with abmil
python downstream_tune.py --model GraphCL --gnn GCN --mode soft --combine_mode abmil --post_mode abmil --dataset Cohort_TCGA_ACC_Phenotype_ajcc_stage --encoder PathOrchestra --encoder_path /mnt/hwfile/smart_health/lujiaxuan/PathOrchestra/eval/weights/PathOrchestra_V1.0.0.bin --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 16 --num_parts 500 --num_workers 0 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/Combined.GraphCL.GCN.PathOrchestra_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_18_loss_2.3723.pth --checkpoint_suffix ABMIL_UNI_Combined.GraphCL.GCN.PathOrchestra_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_18_loss_2.3723

# patch encoder: PathOrchestra, slide encoder: Random-GNN
# freeze gcn, tuning cls head with abmil
python downstream_tune.py --model GraphCL --gnn GCN --mode soft-random --combine_mode abmil --post_mode abmil --dataset Cohort_TCGA_ACC_Phenotype_ajcc_stage --encoder PathOrchestra --encoder_path /mnt/hwfile/smart_health/lujiaxuan/PathOrchestra/eval/weights/PathOrchestra_V1.0.0.bin --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 16 --num_parts 500 --num_workers 0 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/WSIGraph/pre_trained_gnn/Combined.GraphCL.GCN.PathOrchestra_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_18_loss_2.3723.pth --checkpoint_suffix ABMIL_UNI_Combined.GraphCL.GCN.PathOrchestra_light_GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32_epoch_18_loss_2.3723


# -----old, need modify-----------
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