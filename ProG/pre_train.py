import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchmetrics
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from random import shuffle
import random

from ProG.prompt import GNN, ClusterGNN, SoftClusterGNN
from ProG.gcn import GcnEncoderGraph
from ProG.encoder.patch_encoder import PatchEncoder
from ProG.utils import gen_ran_output,load_data4pretrain,mkdir,collate_fn,get_model
from ProG.graph import graph_views, graph_views_ori

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphCL(torch.nn.Module):

    def __init__(self, gnn, hid_dim=16):
        super(GraphCL, self).__init__()
        self.gnn = gnn
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(hid_dim, hid_dim))

    def forward_cl(self, x, edge_index, batch, coord=None):
        x = self.gnn(x, edge_index, batch, coord)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        loss = - torch.log(loss).mean() + 10
        return loss


class PreTrain(torch.nn.Module):
    def __init__(self, pretext="GraphCL", gnn_type='GCN', encoder='Pathoduet', encoder_path=None,
                 gln=2, cluster_sizes=[100, 50, 10], num_workers=1, mode='original'):
        super(PreTrain, self).__init__()
        self.pretext = pretext
        self.gnn_type = gnn_type
        self.num_workers = num_workers

        self.encoder=encoder
        self.enc=PatchEncoder(encoder, encoder_path)

        # pass: get the input_dim and hid_dim for each encoder
        if encoder=='Pathoduet':
            input_dim, hid_dim = 768, 768
        elif encoder=='ResNet50':
            input_dim, hid_dim = 1024, 1024
        elif encoder=='ResNet18':
            input_dim, hid_dim = 256,256
        else:
            input_dim, hid_dim = 100, 100

        if mode == 'original':
            # the original GNN which do not contain pooling
            self.gnn = GNN(input_dim=input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=gln, pool=None,
                           gnn_type=gnn_type)
        elif mode == 'hard': # memory consuming
            self.gnn = ClusterGNN(input_dim=input_dim, hid_dim=hid_dim, out_dim=hid_dim,
                                  cluster_sizes=cluster_sizes, pool=None, gnn_type=gnn_type)
        elif mode == 'soft': # use GNN to learn the cluster id
            self.gnn = SoftClusterGNN(input_dim=input_dim, hid_dim=hid_dim, out_dim=hid_dim,
                                  cluster_sizes=cluster_sizes, pool=None, gnn_type=gnn_type)

        # if self.enc is not None:
        #     # self.enc.to(device)
        #     self.enc.eval()
        # # self.gnn.to(device)

        if pretext in ['GraphCL', 'SimGRACE']:
            self.model = GraphCL(self.gnn, hid_dim=hid_dim)
            # self.model.to(device)
        else:
            raise ValueError("pretext should be GraphCL, SimGRACE")

        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model.cuda())
            # self.model = get_model(self.model)
            if self.enc is not None:
                self.enc = nn.DataParallel(self.enc.cuda())
                self.enc.eval()
                # self.enc = get_model(self.enc)



    def get_loader(self, graph_list, batch_size,
                   aug1=None, aug2=None, aug_ratio=None, pretext="GraphCL", dataname=None):

        # if len(graph_list) % batch_size == 1:
        #     raise KeyError(
        #         "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")

        if pretext == 'GraphCL':
            if dataname == 'CiteSeer':
                shuffle(graph_list)


            if dataname in ['Prostate', 'TCGA']:
                loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False,
                                     num_workers=self.num_workers)  # you must set shuffle=False !
                loader.collate_fn = collate_fn
                return loader, None
            else:
                if aug1 is None:
                    aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
                if aug2 is None:
                    aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
                if aug_ratio is None:
                    aug_ratio = random.randint(1, 3) * 1.0 / 10  # 0.1,0.2,0.3

                print("===graph views: {} and {} with aug_ratio: {}".format(aug1, aug2, aug_ratio))

                view_list_1 = []
                view_list_2 = []
                for g in graph_list:
                    view_g = graph_views_ori(data=g, aug=aug1, aug_ratio=aug_ratio)
                    view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
                    view_list_1.append(view_g)
                    view_g = graph_views_ori(data=g, aug=aug2, aug_ratio=aug_ratio)
                    view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
                    view_list_2.append(view_g)

                loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                                     num_workers=self.num_workers)  # you must set shuffle=False !
                loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                     num_workers=self.num_workers)  # you must set shuffle=False !

                return loader1, loader2
        elif pretext == 'SimGRACE':
            loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
            return loader, None  # if pretext==SimGRACE, loader2 is None
        else:
            raise ValueError("pretext should be GraphCL, SimGRACE")

    def train_simgrace(self, model, loader, optimizer):
        model.train()
        train_loss_accum = 0
        total_step = 0
        for step, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.cuda()
            x2 = gen_ran_output(data, model) 
            x1 = model.module.forward_cl(data.x, data.edge_index, data.batch)
            x2 = Variable(x2.detach().data.cuda(), requires_grad=False)
            loss = model.module.loss_cl(x1, x2)
            loss.backward()
            optimizer.step()
            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def train_graphcl_ori(self, model, loader1, loader2, optimizer):
        model.train()
        train_loss_accum = 0
        total_step = 0
        for step, batch in enumerate(zip(loader1, loader2)):
            batch1, batch2 = batch
            optimizer.zero_grad()
            batch1.x, batch1.edge_index, batch1.batch = batch1.x.cuda(), batch1.edge_index.cuda(), batch1.batch.cuda()
            batch2.x, batch2.edge_index, batch2.batch = batch2.x.cuda(), batch2.edge_index.cuda(), batch2.batch.cuda()

            x1 = model.module.forward_cl(batch1.x, batch1.edge_index, batch1.batch)
            x2 = model.module.forward_cl(batch2.x, batch2.edge_index, batch2.batch)
            loss = model.module.loss_cl(x1, x2)

            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def train_graphcl(self, model, loader1, loader2, optimizer):
        model.train()
        train_loss_accum = 0
        total_step = 0
        for step, batch in enumerate(zip(loader1)):
            # batch1, batch2 = batch[0]
            batch1, batch2, coord_1, coord_2 = batch[0]
            optimizer.zero_grad()

            batch1.x, batch1.edge_index, batch1.batch = batch1.x.cuda(), batch1.edge_index.cuda(), batch1.batch.cuda()
            batch2.x, batch2.edge_index, batch2.batch = batch2.x.cuda(), batch2.edge_index.cuda(), batch2.batch.cuda()

            # get the patch embedding
            if self.enc is not None:
                if self.encoder == 'Pathoduet':
                    batch1.x = self.enc(batch1.x)[0][:, 2:].mean(dim=1)
                    batch2.x = self.enc(batch2.x)[0][:, 2:].mean(dim=1)
                elif self.encoder in ['ResNet50', 'ResNet18']:
                    batch1.x = self.enc(batch1.x)
                    batch2.x = self.enc(batch2.x)

            x1 = model.module.forward_cl(batch1.x, batch1.edge_index, batch1.batch, coord_1)
            x2 = model.module.forward_cl(batch2.x, batch2.edge_index, batch2.batch, coord_2)
            loss = model.module.loss_cl(x1, x2)

            loss.backward()
            optimizer.step()

            # print("loss:", float(loss.detach().cpu().item()))
            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def train(self, dataname, graph_list, batch_size=10, aug1='dropN', aug2="permE", aug_ratio=None, lr=0.01,
              decay=0.0001, epochs=100, checkpoint_suffix='', save_epoch=True):

        loader1, loader2 = self.get_loader(graph_list, batch_size, aug1=aug1, aug2=aug2,
                                           pretext=self.pretext, dataname=dataname)
        # print('start training {} | {} | {}...'.format(dataname, pre_train_method, gnn_type))
        # optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=decay)

        checkpoint_dict = dict()
        train_loss_min = 1000000
        for epoch in range(1, epochs + 1):  # 1..100
            if self.pretext == 'GraphCL':
                train_loss = self.train_graphcl(self.model, loader1, loader2, optimizer)
            elif self.pretext == 'SimGRACE':
                train_loss = self.train_simgrace(self.model, loader1, optimizer)
            else:
                raise ValueError("pretext should be GraphCL, SimGRACE")

            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))

            if train_loss_min > train_loss or save_epoch and epoch % 5 == 0:
                train_loss_min = train_loss
                checkpoint_name = "./pre_trained_gnn/{}.{}.{}.{}_epoch_{}_loss_{}.pth".format(dataname, self.pretext, self.gnn_type, checkpoint_suffix, str(epoch), str(round(train_loss_min,4)))
                checkpoint_dict['epoch'] = checkpoint_name

                torch.save(self.model.module.gnn.state_dict(), checkpoint_name)
                # do not use '../pre_trained_gnn/' because hope there should be two folders: (1) '../pre_trained_gnn/'  and (2) './pre_trained_gnn/'
                # only selected pre-trained models will be moved into (1) so that we can keep reproduction
                print("+++model saved ! {}.{}.{}.pth".format(dataname, self.pretext, self.gnn_type))



if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")

    print(device)
    # device = torch.device('cpu')

    mkdir('./pre_trained_gnn/')
    # do not use '../pre_trained_gnn/' because hope there should be two folders: (1) '../pre_trained_gnn/'  and (2) './pre_trained_gnn/'
    # only selected pre-trained models will be moved into (1) so that we can keep reproduction

    # pretext = 'GraphCL' 
    pretext = 'SimGRACE' 
    gnn_type = 'TransformerConv'  
    # gnn_type = 'GAT'
    # gnn_type = 'GCN'
    dataname, num_parts = 'CiteSeer', 200
    graph_list, input_dim, hid_dim = load_data4pretrain(dataname, num_parts)

    pt = PreTrain(pretext, gnn_type, input_dim, hid_dim, gln=2)
    pt.model.to(device) 
    pt.train(dataname, graph_list, batch_size=10, aug1='dropN', aug2="permE", aug_ratio=None,lr=0.01, decay=0.0001,epochs=100)
