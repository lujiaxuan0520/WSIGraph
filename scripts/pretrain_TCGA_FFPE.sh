proxy_off
conda activate allslide
cd /mnt/hwfile/smart_health/lujiaxuan/WSIGraph
nohup python pretrain.py --model GraphCL --gnn GCN --mode soft --dataset TCGA --encoder Pathoduet --encoder_path /mnt/hwfile/smart_health/lujiaxuan/PathoDuet/models/checkpoint_p2.pth --learning_rate 0.0001 --cluster_sizes 200 200 100 100 50 --batch_size 8 --num_parts 500 --num_workers 32 --checkpoint_suffix GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32 > GCN_soft_pool_cluster_200_200_100_100_50_SGD_lr_0.0001_batch_8_worker_32.txt 2>&1 &