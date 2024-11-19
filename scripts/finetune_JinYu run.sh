#!/bin/bash
conda activate hest
cd /mnt/hwfile/smart_health/lujiaxuan/WSIGraph

nohup python downstream_tune.py --model GraphCL --gnn GCN --mode GigaPath --combine_mode hier_mean --post_mode linear_probing --dataset JinYu --encoder GigaPath --encoder_path /mnt/hwfile/smart_health/lujiaxuan/hest/fm_v1/gigapath/pytorch_model.bin --learning_rate 0.001 --cluster_sizes 200 200 100 100 50 --batch_size 2 --num_parts 500 --num_workers 0 --loss WeightedCrossEntropyLoss --gnn_ckpt /mnt/hwfile/smart_health/lujiaxuan/prov-gigapath/slide_encoder.pth --checkpoint_suffix LP_GigaPath_LongNet_SGD_lr_0.001_batch_8_worker_32 > post-LP_GigaPath_LongNet-4cards_lr0.001-batch4_JinYu.txt 2>&1 &

