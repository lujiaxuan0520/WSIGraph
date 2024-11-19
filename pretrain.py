from ProG.utils import seed_everything, seed

seed_everything(seed)

import argparse
from ProG import PreTrain
from ProG.utils import mkdir, load_data4pretrain
from ProG.prompt import GNN, LightPrompt, HeavyPrompt
from torch import nn, optim
from ProG.data import multi_class_NIG
import torch
from torch_geometric.loader import DataLoader
from ProG.eva import acc_f1_over_batches



def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GraphCL', type=str) # 'GraphCL', 'SimGRACE'

    parser.add_argument('--gnn', default='TransformerConv', type=str) # 'GAT', 'GCN'
    parser.add_argument('--mode', default='hard', type=str)  # 'original', 'hard', 'soft'
    parser.add_argument('--layer', default=2, type=int) # the layer of GNN
    parser.add_argument('--cluster_sizes', nargs='+', type=int, help='cluster size for each layer') # 100,50,10
    parser.add_argument('--dataset', default='CiteSeer', type=str) # 'CiteSeer', 'prostate'
    parser.add_argument('--encoder', default=None, type=str)  # 'Pathoduet', 'ResNet50', 'None'
    parser.add_argument('--encoder_path', default=None, type=str)
    parser.add_argument('--checkpoint_suffix', default='', type=str) # the suffix of checkpoint name
    parser.add_argument('--resume_ckpt', default=None, type=str) # the checkpoint resume training

    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--num_parts', default=200, type=int)
    parser.add_argument('--num_workers', default=1, type=int)

    args = parser.parse_args()

    return args


# this file can not move in ProG.utils.py because it will cause self-loop import
def model_create(dataname, gnn_type, num_class, task_type='multi_class_classification', tune_answer=False):
    if task_type in ['multi_class_classification', 'regression']:
        input_dim, hid_dim = 100, 100
        lr, wd = 0.001, 0.00001
        tnpc = 100  # token number per class

        # load pre-trained GNN
        gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, gnn_type=gnn_type)
        pre_train_path = './pre_trained_gnn/{}.GraphCL.{}.pth'.format(dataname, gnn_type)
        gnn.load_state_dict(torch.load(pre_train_path))
        print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
        for p in gnn.parameters():
            p.requires_grad = False

        if tune_answer:
            PG = HeavyPrompt(token_dim=input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3)
        else:
            PG = LightPrompt(token_dim=input_dim, token_num_per_group=tnpc, group_num=num_class, inner_prune=0.01)

        opi = optim.Adam(filter(lambda p: p.requires_grad, PG.parameters()),
                         lr=lr,
                         weight_decay=wd)

        if task_type == 'regression':
            lossfn = nn.MSELoss(reduction='mean')
        else:
            lossfn = nn.CrossEntropyLoss(reduction='mean')

        if tune_answer:
            if task_type == 'regression':
                answering = torch.nn.Sequential(
                    torch.nn.Linear(hid_dim, num_class),
                    torch.nn.Sigmoid())
            else:
                answering = torch.nn.Sequential(
                    torch.nn.Linear(hid_dim, num_class),
                    torch.nn.Softmax(dim=1))

            opi_answer = optim.Adam(filter(lambda p: p.requires_grad, answering.parameters()), lr=0.01,
                                    weight_decay=0.00001)
        else:
            answering, opi_answer = None, None
        gnn.to(device)
        PG.to(device)
        return gnn, PG, opi, lossfn, answering, opi_answer
    else:
        raise ValueError("model_create function hasn't supported {} task".format(task_type))




if __name__ == '__main__':
    args = make_parse()
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")


    mkdir('./pre_trained_gnn/')
    # pretext = 'GraphCL'  # 'GraphCL', 'SimGRACE'
    # gnn_type = 'TransformerConv'  # 'GAT', 'GCN'
    # dataname, num_parts, batch_size = 'CiteSeer', 200, 10
    pretext = args.model
    gnn_type = args.gnn
    encoder = args.encoder
    encoder_path = args.encoder_path
    dataname, num_parts, batch_size = args.dataset, args.num_parts, args.batch_size

    print("load data...")
    graph_list = load_data4pretrain(dataname, num_parts, encoder=encoder)

    print("create PreTrain instance...")
    pt = PreTrain(pretext, gnn_type, encoder, encoder_path, gln=args.layer, cluster_sizes=args.cluster_sizes,
                  mode=args.mode, num_workers=args.num_workers, resume_ckpt=args.resume_ckpt)

    print("pre-training...")
    pt.train(dataname, graph_list, batch_size=batch_size,
             aug1='dropN', aug2="permE", aug_ratio=None, lr=args.learning_rate, decay=0.0001, epochs=100,
             checkpoint_suffix=args.checkpoint_suffix, save_epoch=True)
