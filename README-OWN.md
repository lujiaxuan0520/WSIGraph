# 使用说明

- 基于base环境，环境问题需要确保以下包都已经安装，后面链接需要写torch版本和cuda版本：特别是需要安装pyg_lib

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
``` 

- torch-geometric文档链接：https://pytorch-geometric.readthedocs.io/en/latest/index.html

## 运行：

- 运行命令：

```bash
python pretrain.py --model GraphCL --gnn GCN --dataset CiteSeer --encoder Pathoduet --encoder_path /mnt/data/smart_health_02/lujiaxuan/workingplace/GleasonGrade/code/PathoDuet/models/checkpoint_p2.pth --batch_size 10 --num_parts 200
```

- 初始化图可视化：参考visualize_graph()