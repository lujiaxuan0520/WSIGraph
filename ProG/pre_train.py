import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchmetrics
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from random import shuffle
import random
import os
import io
from tqdm import tqdm
from petrel_client.client import Client
from torchvision import transforms
import torch.nn.functional as F

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
                 gln=2, cluster_sizes=[100, 50, 10], num_workers=1, mode='original', resume_ckpt=None):
        super(PreTrain, self).__init__()
        self.pretext = pretext
        self.gnn_type = gnn_type
        self.num_workers = num_workers

        self.encoder=encoder
        self.enc=PatchEncoder(encoder, encoder_path)

        self.client = Client('/mnt/petrelfs/yanfang/.petreloss.conf')

        # pass: get the input_dim and hid_dim for each encoder
        if encoder=='Pathoduet':
            input_dim, hid_dim = 768, 768
        elif encoder=='GigaPath':
            input_dim, hid_dim = 1536, 1536
        elif encoder=='UNI':
            input_dim, hid_dim = 1024, 1024
        elif encoder=='PathOrchestra':
            input_dim, hid_dim = 1024, 1024
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

        if resume_ckpt is not None:
            self.gnn.load_state_dict(torch.load(resume_ckpt))
            print("successfully load pre-trained weights for gnn! @ {}".format(resume_ckpt))

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
                   aug1=None, aug2=None, aug_ratio=None, pretext="GraphCL", dataname=None, encoder=None):

        # if len(graph_list) % batch_size == 1:
        #     raise KeyError(
        #         "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")

        if pretext == 'GraphCL':
            if dataname == 'CiteSeer':
                shuffle(graph_list)


            if dataname in ['Prostate', 'TCGA', 'TCGA_frozen', 'RUIJIN', 'RJ_lymphoma', 'Digest_all', 'Tsinghua', 'XIJING', 'IHC', 'Combined']:
                loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False,
                                     num_workers=self.num_workers)  # you must set shuffle=False !
                loader.collate_fn = collate_fn
                return loader, None
            else:
                print("the dataset not supported !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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
                                     num_workers=self.num_workers, encoder=encoder)  # you must set shuffle=False !
                loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                     num_workers=self.num_workers, encoder=encoder)  # you must set shuffle=False !

                return loader1, loader2
        elif pretext == 'SimGRACE':
            loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
            return loader, None  # if pretext==SimGRACE, loader2 is None
        else:
            raise ValueError("pretext should be GraphCL, SimGRACE")

    def upload_tensor_to_ceph(self, tensor, remote_path):
        """
        上传内存中的张量数据到 Ceph
        :param tensor: 待上传的张量
        :param remote_path: Ceph 远程路径
        """
        try:
            # 使用 BytesIO 来模拟文件对象
            with io.BytesIO() as buffer:
                torch.save(tensor, buffer)  # 将张量保存到内存中的 buffer
                buffer.seek(0)  # 重置 buffer 的指针
                self.client.put(remote_path, buffer)  # 上传到 Ceph
            print(f"Tensor data uploaded to {remote_path} successfully.")
        except Exception as e:
            print(f"Error uploading tensor to {remote_path}: {e}")

    def load_tensor_from_ceph(self, file_path):
        try:
            data_bytes = self.client.get(file_path)
            if data_bytes is None:
                raise ValueError(f"Failed to load {file_path} from Ceph: Data is None")
            tensor = torch.load(io.BytesIO(data_bytes))
            return tensor
        except Exception as e:
            print(f"Error loading tensor from {file_path}: {e}")
            raise e

    # check whether the path exits (only for files on the ceph)
    def path_exists(self, paths):
        """检查路径是否存在"""
        try:
            return all(self.client.contains(path) for path in paths)
        except:
            return False

    # 定义处理单个 batch 的函数
    def process_batch(self, batch_x, batch_indices, file_paths, view):
        unique_batches = batch_indices.unique()
        processed_embeddings = []
        new_batch = []

        for batch_id in unique_batches:
            # Get the indices for the current batch_id
            mask = (batch_indices == batch_id)
            x_subset = batch_x[mask]
            file_path = file_paths[batch_id.item()]

            file_path = file_path.replace("/data/", "/features/")
            file_path = os.path.join(file_path, self.encoder + '_' + view + '.pt')

            if self.path_exists([file_path]):
                try:
                    # Load the embedding from Ceph
                    embedding = self.load_tensor_from_ceph(file_path).cuda()
                except Exception as e:
                    print(f"Error loading {file_path} from Ceph: {e}")
                    # If loading fails, compute and upload the embedding
                    embedding = self.compute_and_upload_embedding(x_subset, file_path, view)
            else:
                # Compute and upload the embedding
                embedding = self.compute_and_upload_embedding(x_subset, file_path, view)

            

            ori_batch = batch_indices[mask]
            if embedding.shape[0] != ori_batch.shape[0]:
                print("!!!Error dimension for reading file:", file_path)
                # modify the information for consistent dimension
                # # tmp_batch = torch.full((embedding.shape[0],), batch_id)
                # # new_batch.append(tmp_batch.cuda())
                # rows_to_add = ori_batch.shape[0] - embedding.shape[0]
                # zero_rows = torch.zeros((rows_to_add, embedding.shape[1])).cuda()
                # indices = torch.randperm(embedding.shape[0] + rows_to_add).cuda()
                # augmented_embedding = torch.cat([embedding, zero_rows], dim=0)
                # embedding = augmented_embedding[indices]
                ori_batch_size = ori_batch.shape[0]
                embedding_size = embedding.shape[0]
                
                if ori_batch_size > embedding_size:
                    # Case 1: ori_batch has more samples than embedding
                    rows_to_add = ori_batch_size - embedding_size
                    zero_rows = torch.zeros((rows_to_add, embedding.shape[1])).cuda()
                    augmented_embedding = torch.cat([embedding, zero_rows], dim=0)
                    indices = torch.randperm(augmented_embedding.shape[0]).cuda()
                    embedding = augmented_embedding[indices]
                elif ori_batch_size < embedding_size:
                    # Case 2: ori_batch has fewer samples than embedding
                    rows_to_remove = embedding_size - ori_batch_size
                    indices = torch.randperm(embedding_size).cuda()
                    selected_indices = indices[:ori_batch_size]
                    embedding = embedding[selected_indices]
                
            new_batch.append(ori_batch)
            processed_embeddings.append(embedding)
                
        # if embedding.shape[0] != ori_batch.shape[0]:
        #     batch_indices = torch.cat(new_batch, dim=0)

        # Concatenate all embeddings back into a single tensor
        return torch.cat(processed_embeddings, dim=0), torch.cat(new_batch, dim=0)

    # Define a helper function to compute and upload embeddings
    def compute_and_upload_embedding(self, x_subset, file_path, view):
        # Move the subset to GPU if not already
        x_subset = x_subset.cuda()

        # Compute the embedding based on the encoder type
        if self.enc is not None:
            if self.encoder == 'Pathoduet':
                enc_output = self.enc(x_subset)[0][:, 2:].mean(dim=1)
            elif self.encoder in ['ResNet50', 'ResNet18']:
                enc_output = self.enc(x_subset)
            elif self.encoder == 'GigaPath':
                enc_output = self.enc(x_subset)
            elif self.encoder == 'UNI':
                # transform = transforms.Compose(
                #     [
                #         transforms.Resize(224),
                #         transforms.ToTensor(),
                #         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                #     ]
                # )
                x_subset = F.interpolate(x_subset, size=(224, 224), mode="bilinear", align_corners=False)
                mean = torch.tensor([0.485, 0.456, 0.406], dtype=x_subset.dtype, device=x_subset.device)
                std = torch.tensor([0.229, 0.224, 0.225], dtype=x_subset.dtype, device=x_subset.device)
                x_subset = (x_subset - mean[None, :, None, None]) / std[None, :, None, None]
                enc_output = self.enc(x_subset)
            elif self.encoder=='PathOrchestra':
                enc_output = self.enc(x_subset)
            else:
                raise ValueError(f"Unsupported encoder type: {self.encoder}")

            embedding = enc_output.detach()  # Detach if gradient is not needed
        else:
            raise ValueError("Encoder (self.enc) is not defined.")

        # Upload the embedding to Ceph
        try:
            self.upload_tensor_to_ceph(embedding.cpu(), file_path)
        except Exception as e:
            print(f"Error uploading {file_path} to Ceph: {e}")

        return embedding

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

    # def train_graphcl(self, model, loader1, loader2, optimizer):
    #     model.train()
    #     train_loss_accum = 0
    #     total_step = 0
    #     for step, batch in enumerate(loader1):
    #     # for step, batch in enumerate(zip(loader1)):
    #     # for step, batch in enumerate(tqdm(loader1, desc="Training Progress")):
    #         # batch1, batch2 = batch[0]
    #         # print("Step:", step)
    #         batch1, batch2, coord_1, coord_2 = batch[0]
    #         file_paths = batch[1]
    #         optimizer.zero_grad()

    #         batch1.x, batch1.edge_index, batch1.batch = batch1.x.cuda(), batch1.edge_index.cuda(), batch1.batch.cuda()
    #         batch2.x, batch2.edge_index, batch2.batch = batch2.x.cuda(), batch2.edge_index.cuda(), batch2.batch.cuda()

    #         # get the patch embedding
    #         if self.enc is not None:
    #             if self.encoder == 'Pathoduet':
    #                 batch1.x = self.enc(batch1.x)[0][:, 2:].mean(dim=1)
    #                 batch2.x = self.enc(batch2.x)[0][:, 2:].mean(dim=1)
    #             elif self.encoder in ['ResNet50', 'ResNet18']:
    #                 batch1.x = self.enc(batch1.x)
    #                 batch2.x = self.enc(batch2.x)
    #             elif self.encoder=='GigaPath':
    #                 # self.enc.eval()
    #                 # with torch.no_grad():
    #                     batch1.x = self.enc(batch1.x)
    #                     batch2.x = self.enc(batch2.x)

    #         x1 = model.module.forward_cl(batch1.x, batch1.edge_index, batch1.batch, coord_1)
    #         x2 = model.module.forward_cl(batch2.x, batch2.edge_index, batch2.batch, coord_2)
    #         loss = model.module.loss_cl(x1, x2)

    #         loss.backward()
    #         optimizer.step()

    #         # print("loss:", float(loss.detach().cpu().item()))
    #         train_loss_accum += float(loss.detach().cpu().item())
    #         total_step = total_step + 1

    #     return train_loss_accum / total_step

    def train_graphcl(self, model, loader1, loader2, optimizer):
        model.train()
        train_loss_accum = 0
        total_step = 0
        for step, batch in enumerate(loader1):
        # for step, batch in enumerate(zip(loader1)):
        # for step, batch in enumerate(tqdm(loader1, desc="Training Progress")):
            # batch1, batch2 = batch[0]
            # print("Step:", step)
            batch1, batch2, coord_1, coord_2 = batch[0]
            file_paths = batch[1]
            optimizer.zero_grad()

            batch1.x, batch1.edge_index, batch1.batch = batch1.x.cuda(), batch1.edge_index.cuda(), batch1.batch.cuda()
            batch2.x, batch2.edge_index, batch2.batch = batch2.x.cuda(), batch2.edge_index.cuda(), batch2.batch.cuda()
            coord_1, coord_2 = coord_1.cuda(), coord_2.cuda()


            # # get the patch embedding
            # if self.enc is not None:
            #     if self.encoder == 'Pathoduet':
            #         batch1.x = self.enc(batch1.x)[0][:, 2:].mean(dim=1)
            #         batch2.x = self.enc(batch2.x)[0][:, 2:].mean(dim=1)
            #     elif self.encoder in ['ResNet50', 'ResNet18']:
            #         batch1.x = self.enc(batch1.x)
            #         batch2.x = self.enc(batch2.x)
            #     elif self.encoder=='GigaPath':
            #         # self.enc.eval()
            #         # with torch.no_grad():
            #             batch1.x = self.enc(batch1.x)
            #             batch2.x = self.enc(batch2.x)

            batch1_embeddings, batch1.batch = self.process_batch(batch1.x, batch1.batch, file_paths, view="view1")
            batch2_embeddings, batch2.batch = self.process_batch(batch2.x, batch2.batch, file_paths, view="view2")
            batch1.x = batch1_embeddings
            batch2.x = batch2_embeddings

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
                                           pretext=self.pretext, dataname=dataname, encoder=self.encoder)
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

            # if train_loss_min > train_loss:
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
