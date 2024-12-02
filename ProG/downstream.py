import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchmetrics
from torchmetrics.classification import Recall, Precision, Specificity
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torchvision import transforms
import torch.nn.functional as F
import torch.nn.functional as F
from random import shuffle
import random
import math

import sys
sys.path.append('/mnt/hwfile/smart_health/lujiaxuan/prov-gigapath')
from gigapath.slide_encoder import create_model

from ProG.prompt import GNN, ClusterGNN, SoftClusterGNN
from ProG.gcn import GcnEncoderGraph
from ProG.encoder.patch_encoder import PatchEncoder
from ProG.utils import gen_ran_output,load_data4pretrain,mkdir,collate_fn,get_model
from ProG.graph import graph_views, graph_views_ori
from ProG.loss import SlideOnlyCriterion
from ProG.abmil import AttnMIL6, Classifier_1fc

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class GraphCL(torch.nn.Module):
#
#     def __init__(self, gnn, hid_dim=16):
#         super(GraphCL, self).__init__()
#         self.gnn = gnn
#         self.projection_head = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
#                                                    torch.nn.ReLU(inplace=True),
#                                                    torch.nn.Linear(hid_dim, hid_dim))
#
#     def forward_cl(self, x, edge_index, batch, coord=None):
#         x = self.gnn(x, edge_index, batch, coord)
#         x = self.projection_head(x)
#         return x
#
#     def loss_cl(self, x1, x2):
#         T = 0.1
#         batch_size, _ = x1.size()
#         x1_abs = x1.norm(dim=1)
#         x2_abs = x2.norm(dim=1)
#         sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
#         sim_matrix = torch.exp(sim_matrix / T)
#         pos_sim = sim_matrix[range(batch_size), range(batch_size)]
#         loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
#         loss = - torch.log(loss).mean() + 10
#         return loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageAUROC(torchmetrics.AUROC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "AverageAUROC"

class PerClassAUROC(torchmetrics.AUROC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "PerClassAUROC"

class FT(torch.nn.Module):

    def __init__(self, gnn, hid_dim=16, combine_mode='graph_level', post_mode=None,
        class_num=10, loss_name='WeightedCrossEntropyLoss'):
        '''
        :param gnn:
        :param hid_dim:
        :param combine_mode:
            graph_level: use only graph level repr;
            region_level: use only region level repr by mean;
            hier_mean: hierarchically combine graph, region and node level reprs by mean;
        '''
        super(FT, self).__init__()
        self.combine_mode = combine_mode
        self.post_mode = post_mode
        self.gnn = gnn
        self.loss_name = loss_name
        self.class_num = class_num

        self.layer_norm = nn.LayerNorm(normalized_shape=hid_dim)


        if self.combine_mode == 'hier_weighted_mean':
            self.weights = nn.Parameter(torch.ones(len(self.gnn.cluster_sizes), 1, hid_dim))
            nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        
        # self.projection_head = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
        #                                            torch.nn.ReLU(inplace=True),
        #                                            torch.nn.Linear(hid_dim, hid_dim))

        if self.post_mode in ['linear_probing', 'abmil']:
            # freeze the gnn
            for param in self.gnn.parameters():
                param.requires_grad = False

        # cls head
        # self.cls_head = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
        #                                     torch.nn.ReLU(inplace=True),
        #                                     torch.nn.Linear(hid_dim, class_num),
        #                                     # torch.nn.LayerNorm(class_num)
        #                                     torch.nn.BatchNorm1d(class_num)
        #                                     )

        if self.post_mode == 'abmil':
            class Config:
                D_feat = hid_dim  # Number of features in the input instances
                D_inner = 128  # Reduced feature dimension
                n_token = 1    # Number of attention branches (or tokens)
                n_class = class_num    # Number of classes for classification
                n_masked_patch = 0  # Number of masked patches, set to 0 if not used
                mask_drop = 0.5  # Proportion of masks to drop if n_masked_patch > 0
            conf = Config()
            self.abmil = AttnMIL6(conf)
            for m in self.abmil.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            self.cls_head = Classifier_1fc(128, class_num, 0.5)
            for m in self.cls_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
        else:
            self.cls_head = torch.nn.Sequential(torch.nn.Linear(hid_dim, class_num),
                                                # torch.nn.LayerNorm(class_num)
                                                torch.nn.BatchNorm1d(class_num)
                                                )
            for m in self.cls_head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)


    def forward_ft(self, x, edge_index, batch, coord=None):
        print("x:", x.shape)
        print("edge_index:", edge_index.shape)
        print("batch:", batch.shape)
        print("coord:", coord.shape)
        if int(x.shape[0]) == 2994:
            print('debug')
        batch_size = batch.max().item() + 1
        abmil = True if self.combine_mode == 'abmil' else False
        
        if self.combine_mode == 'linear_probing':
            sample_x = []
            for i in range(batch_size):
                batch_elements = x[batch == i]
                if batch_elements.size(0) > 0:
                    x_mean = torch.mean(batch_elements, dim=0)
                    sample_x.append(x_mean)
            sample_x = torch.stack(sample_x)
            return self.cls_head(sample_x)

        # for GigaPath-Slide encoder
        if hasattr(self.gnn, 'encoder_name') and 'LongNet' in self.gnn.encoder_name:
            unique_batches = torch.unique(batch)
            all_embeddings = []
            # all_preds = []

            self.gnn.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    coord = coord.to(x.device)
                    for b in unique_batches:
                        mask = (batch == b)
                        tile_embed = x[mask].unsqueeze(0)
                        coords = coord[mask].unsqueeze(0)
                        
                        if abmil:
                            output, features = self.gnn(tile_embed, coords, abmil=True)
                            # class_output, slide_output, attention_weights = self.cls_head(features[0])
                            output = self.abmil(features[0])
                            # all_preds.append(class_output)
                            all_embeddings.append(output)
                        else:
                            output = self.gnn(tile_embed, coords)
                            all_embeddings.append(output[0])

            final_embedding = torch.cat(all_embeddings, dim=0)
            output = self.cls_head(final_embedding.float())
            return output
        

        x, vertex_features, cluster_results = self.gnn(x, edge_index, batch, coord)
        # x = self.projection_head(x)

        if self.combine_mode == 'hier_mean':
            sample_averages = []
            for i in range(batch_size):
                sample_feature_sum = torch.zeros(1, vertex_features[0].shape[-1]).to(x.device)

                for features in vertex_features:
                    n = features.shape[0] // batch_size
                    sample_features = features[i * n:(i + 1) * n]
                    layer_mean = torch.mean(sample_features, dim=0, keepdim=True).to(x.device)
                    layer_mean = self.layer_norm(layer_mean)
                    sample_feature_sum += layer_mean

                sample_average = sample_feature_sum / len(vertex_features)
                sample_averages.append(sample_average)

            x = torch.cat(sample_averages, dim=0)
        elif self.combine_mode == 'hier_weighted_mean':
            sample_averages = []
            batch_size = batch.max().item() + 1
            for i in range(batch_size):
                sample_feature_sum = torch.zeros(1, vertex_features[0].shape[-1]).to(x.device)

                for idx, features in enumerate(vertex_features):
                    n = features.shape[0] // batch_size
                    sample_features = features[i * n:(i + 1) * n]
                    layer_mean = torch.mean(sample_features, dim=0, keepdim=True).to(x.device)
                    layer_mean = self.layer_norm(layer_mean)
                    weighted_mean = layer_mean * self.weights[idx]
                    sample_feature_sum += weighted_mean

                sample_average = sample_feature_sum / len(vertex_features)
                sample_averages.append(sample_average)

            x = torch.cat(sample_averages, dim=0)
        elif self.combine_mode == 'region_level':
            sample_averages = []
            batch_size = batch.max().item() + 1
            layer = -1 # only use the last hierarchical layer
            for i in range(batch_size):
                n = vertex_features[layer].shape[0] // batch_size
                sample_features = vertex_features[layer][i * n:(i + 1) * n]
                sample_average = torch.mean(sample_features, dim=0, keepdim=True).to(x.device)
                sample_averages.append(sample_average)
            x = torch.cat(sample_averages, dim=0)
        elif self.combine_mode == 'graph_level':
            pass
        elif self.combine_mode == 'abmil':
            sample_averages = []
            batch_size = batch.max().item() + 1
            layer = -1 # only use the last hierarchical layer
            for i in range(batch_size):
                n = vertex_features[layer].shape[0] // batch_size
                sample_features = vertex_features[layer][i * n:(i + 1) * n]
                output = self.abmil(sample_features.unsqueeze(0))
                # all_preds.append(class_output)
                sample_averages.append(output)
            x = torch.cat(sample_averages, dim=0)

        # cls head
        x = self.cls_head(x)

        return x


    def loss_ft(self, x, labels):
        # return nn.CrossEntropyLoss()(x, labels)
        return SlideOnlyCriterion(device='cuda', loss_name=self.loss_name)(x, labels)

class Downstream(torch.nn.Module):
    def __init__(self, pretext="GraphCL", gnn_type='GCN', encoder='Pathoduet', encoder_path=None,
                 gln=2, cluster_sizes=[100, 50, 10], num_workers=1, mode='original', combine_mode='graph_level', post_mode=None, 
                 class_num=10, loss_name='WeightedCrossEntropyLoss', gnn_ckpt=None):
        super(Downstream, self).__init__()
        self.pretext = pretext
        self.gnn_type = gnn_type
        self.num_workers = num_workers
        self.loss_name = loss_name
        self.class_num = class_num

        self.encoder=encoder
        self.enc=PatchEncoder(encoder, encoder_path)

        self.configure_loss_metric()

        # pass: get the input_dim and hid_dim for each encoder
        if encoder=='Pathoduet':
            input_dim, hid_dim = 768, 768
        elif encoder=='GigaPath':
            # input_dim, hid_dim = 1536, 1536
            input_dim, hid_dim = 768, 768
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
            self.gnn.load_state_dict(torch.load(gnn_ckpt))
        elif mode == 'hard': # memory consuming
            self.gnn = ClusterGNN(input_dim=input_dim, hid_dim=hid_dim, out_dim=hid_dim,
                                  cluster_sizes=cluster_sizes, pool=None, gnn_type=gnn_type)
            self.gnn.load_state_dict(torch.load(gnn_ckpt))
        elif mode == 'soft': # use GNN to learn the cluster id
            if encoder=='GigaPath': # temp change the dimension
                input_dim, hid_dim = 1536, 1536
            elif encoder=='UNI':
                input_dim, hid_dim = 1024, 1024
            elif encoder=='PathOrchestra':
                input_dim, hid_dim = 1024, 1024
            self.gnn = SoftClusterGNN(input_dim=input_dim, hid_dim=hid_dim, out_dim=hid_dim,
                                  cluster_sizes=cluster_sizes, pool=None, gnn_type=gnn_type, phase='finetune')
            self.gnn.load_state_dict(torch.load(gnn_ckpt))
        elif mode == 'GigaPath': # use GigaPath-Slide encoder
            self.gnn = slide_encoder = create_model(gnn_ckpt, "gigapath_slide_enc12l768d", 1536)

        # if self.enc is not None:
        #     # self.enc.to(device)
        #     self.enc.eval()
        # # self.gnn.to(device)

        
        print("successfully load pre-trained weights for gnn! @ {}".format(gnn_ckpt))

        if pretext in ['GraphCL', 'SimGRACE']:
            self.model = FT(self.gnn, hid_dim=hid_dim, combine_mode=combine_mode, post_mode=post_mode, class_num=class_num, loss_name=self.loss_name)
            # self.model.to(device)
        else:
            raise ValueError("pretext should be GraphCL, SimGRACE")

        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model.cuda())
            # self.model = get_model(self.model)
            if self.enc is not None:
                # if encoder=='GigaPath':
                #     self.enc = self.enc.cuda()
                # else:
                self.enc = nn.DataParallel(self.enc.cuda())
                self.enc.eval()
                # self.enc = get_model(self.enc)



    def get_loader(self, graph_list, batch_size, pretext="GraphCL", dataname=None):

        # if len(graph_list) % batch_size == 1:
        #     raise KeyError(
        #         "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")

        if pretext == 'GraphCL':
            if dataname == 'CiteSeer':
                shuffle(graph_list)


            if dataname in ['Prostate', 'TCGA', 'JinYu']:
                loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False,
                                     num_workers=self.num_workers, drop_last=True)  # you must set shuffle=False !
                loader.collate_fn = collate_fn
                return loader, None
            else:
                # if aug1 is None:
                #     aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
                # if aug2 is None:
                #     aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
                # if aug_ratio is None:
                #     aug_ratio = random.randint(1, 3) * 1.0 / 10  # 0.1,0.2,0.3
                #
                # print("===graph views: {} and {} with aug_ratio: {}".format(aug1, aug2, aug_ratio))

                view_list = []
                for g in graph_list:
                    # view_g = graph_views_ori(data=g, aug=aug1, aug_ratio=aug_ratio)
                    view_g = Data(x=g.x, edge_index=g.edge_index)
                    view_list.append(view_g)

                loader = DataLoader(view_list, batch_size=batch_size, shuffle=False,
                                     num_workers=self.num_workers, drop_last=True)  # you must set shuffle=False !

                return loader, None
        elif pretext == 'SimGRACE':
            loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
            return loader, None  # if pretext==SimGRACE, loader2 is None
        else:
            raise ValueError("pretext should be GraphCL, SimGRACE")

    # def train_simgrace(self, model, loader, optimizer):
    #     model.train()
    #     train_loss_accum = 0
    #     total_step = 0
    #     for step, data in enumerate(loader):
    #         optimizer.zero_grad()
    #         data = data.cuda()
    #         x2 = gen_ran_output(data, model)
    #         x1 = model.module.forward_cl(data.x, data.edge_index, data.batch)
    #         x2 = Variable(x2.detach().data.cuda(), requires_grad=False)
    #         loss = model.module.loss_cl(x1, x2)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss_accum += float(loss.detach().cpu().item())
    #         total_step = total_step + 1
    #
    #     return train_loss_accum / total_step

    # def train_graphcl_ori(self, model, loader1, loader2, optimizer):
    #     model.train()
    #     train_loss_accum = 0
    #     total_step = 0
    #     for step, batch in enumerate(zip(loader1, loader2)):
    #         batch1, batch2 = batch
    #         optimizer.zero_grad()
    #         batch1.x, batch1.edge_index, batch1.batch = batch1.x.cuda(), batch1.edge_index.cuda(), batch1.batch.cuda()
    #         batch2.x, batch2.edge_index, batch2.batch = batch2.x.cuda(), batch2.edge_index.cuda(), batch2.batch.cuda()
    #
    #         x1 = model.module.forward_cl(batch1.x, batch1.edge_index, batch1.batch)
    #         x2 = model.module.forward_cl(batch2.x, batch2.edge_index, batch2.batch)
    #         loss = model.module.loss_cl(x1, x2)
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         train_loss_accum += float(loss.detach().cpu().item())
    #         total_step = total_step + 1
    #
    #     return train_loss_accum / total_step


    def configure_loss_metric(self):
        # loss = self.cfg.Loss.baseloss

        # torch metric
        # metrics = torchmetrics.MetricCollection([torchmetrics.AUROC(
        #     num_classes=self.cfg.Model.num_classes, average='macro')]) # average AUROC
        try:
            metrics = torchmetrics.MetricCollection([
                # AUC
                AverageAUROC(num_classes=self.class_num, average='macro'),
                PerClassAUROC(num_classes=self.class_num, average=None),
                # F1 Score
                torchmetrics.F1(num_classes=self.class_num, average='macro'),
                # Accuracy
                torchmetrics.Accuracy(),
                Recall(num_classes=self.class_num, average='macro'),
                Precision(num_classes=self.class_num, average='macro'),
                Specificity(num_classes=self.class_num, average='macro')
            ])
        except Exception:
            metrics = torchmetrics.MetricCollection([
                # AUC
                AverageAUROC(num_classes=self.class_num, average='macro'),
                PerClassAUROC(num_classes=self.class_num, average=None),
                # F1 Score
                torchmetrics.F1Score(num_classes=self.class_num, average='macro'),
                # Accuracy
                torchmetrics.Accuracy(),
                Recall(num_classes=self.class_num, average='macro'),
                Precision(num_classes=self.class_num, average='macro'),
                Specificity(num_classes=self.class_num, average='macro')
            ])
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')


    def train_downstream_ft(self, model, loader1, loader2, optimizer):
        model.train()
        train_loss_accum = 0
        total_step = 0
        # for step, batch in enumerate(loader1):
        for step, batch in enumerate(loader1):
        # for step, batch in enumerate(zip(loader1)):
            # batch1, batch2 = batch[0]
            batch, _, coord, _, labels = batch[0]
            optimizer.zero_grad()

            batch.x, batch.edge_index, batch.batch = batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda()

            # get the patch embedding
            if self.enc is not None:
                with torch.no_grad():
                    if self.encoder == 'Pathoduet':
                        batch.x = self.enc(batch.x)[0][:, 2:].mean(dim=1)
                    elif self.encoder in ['ResNet50', 'ResNet18']:
                        batch.x = self.enc(batch.x)
                    elif self.encoder=='GigaPath':
                        # with torch.no_grad():
                        # batch.x = self.enc(batch.x)

                        ## process the big batch by multiple mini batch
                        outputs = []
                        batch_size=10 # process every 200 elements
                        N = batch.x.shape[0]
                        for i in range(0, N, batch_size): 
                            x_batch = batch.x[i:i+batch_size]
                            encoded_batch = self.enc(x_batch)
                            outputs.append(encoded_batch)
                        
                        batch.x = torch.cat(outputs, dim=0)
                    elif self.encoder=='UNI':
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
                        batch.x = self.enc(batch.x)
                    elif self.encoder=='PathOrchestra':
                        batch.x = self.enc(batch.x)

                            

            if labels.dtype != torch.long:
                labels = labels.long()

            x = model.module.forward_ft(batch.x, batch.edge_index, batch.batch, coord)

            loss = model.module.loss_ft(x, labels)[0]

            loss.backward()
            optimizer.step()

            # print("loss:", float(loss.detach().cpu().item()))
            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def test_model(self, model, test_loader):
        model.eval()  # Set the model to evaluation mode
        self.enc.eval()
        self.test_metrics.reset()  # Reset metrics for clean evaluation
        test_loss_accum = 0
        total_step = 0

        with torch.no_grad():  # Disable gradient computation
            for step, batch in enumerate(test_loader):
                batch, _, coord, _, labels = batch

                # Ensure data is on the correct device (cuda in this case)
                batch.x, batch.edge_index, batch.batch = batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda()

                # Get the patch embedding
                if self.enc is not None:
                    if self.encoder == 'Pathoduet':
                        batch.x = self.enc(batch.x)[0][:, 2:].mean(dim=1)
                    elif self.encoder in ['ResNet50', 'ResNet18']:
                        batch.x = self.enc(batch.x)
                    elif self.encoder=='GigaPath':
                        # with torch.no_grad():
                        batch.x = self.enc(batch.x)
                    elif self.encoder=='UNI':
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
                        batch.x = self.enc(batch.x)
                    elif self.encoder=='PathOrchestra':
                        batch.x = self.enc(batch.x)

                # Forward pass
                x = self.model.module.forward_ft(batch.x, batch.edge_index, batch.batch, coord)

                if labels.dtype != torch.long:
                    labels = labels.long()

                # Compute loss
                labels = labels.to(x.device)
                loss = self.model.module.loss_ft(x, labels)
                test_loss_accum += loss[0].item()

                # Update metrics
                self.test_metrics.to(labels.device)
                self.test_metrics.update(x.to(labels.device), labels)
                total_step += 1

        # Compute the final metrics for the entire test set
        test_metrics_results = self.test_metrics.compute()

        # Report average test loss
        average_test_loss = test_loss_accum / total_step
        test_metrics_results['Average Loss'] = average_test_loss

        self.model.train()  # Set the model back to training mode
        return test_metrics_results

    def train(self, dataname, train_graph_list, test_graph_list, batch_size=10, lr=0.01,
              decay=0.0001, epochs=100, checkpoint_suffix='', save_epoch=True):

        train_loader, _ = self.get_loader(train_graph_list, batch_size, pretext=self.pretext, dataname=dataname)
        # val_loader, _ = self.get_loader(val_graph_list, batch_size, pretext=self.pretext, dataname=dataname)
        test_loader, _ = self.get_loader(test_graph_list, batch_size, pretext=self.pretext, dataname=dataname)
        # print('start training {} | {} | {}...'.format(dataname, pre_train_method, gnn_type))
        # optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=decay)

        checkpoint_dict = dict()
        train_loss_min = 1000000
        for epoch in range(1, epochs + 1):  # 1..100
            # if self.pretext == 'GraphCL':
            #     train_loss = self.train_downstream_ft(self.model, loader, _, optimizer)
            # elif self.pretext == 'SimGRACE':
            #     train_loss = self.train_simgrace(self.model, loader, optimizer)
            # else:
            #     raise ValueError("pretext should be GraphCL, SimGRACE")
            train_loss = self.train_downstream_ft(self.model, train_loader, _, optimizer)

            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                # checkpoint_name = "./pre_trained_gnn/{}.{}.{}.{}_epoch_{}_loss_{}.pth".format(dataname, self.pretext, self.gnn_type, checkpoint_suffix, str(epoch), str(round(train_loss_min,4)))
                # checkpoint_dict['epoch'] = checkpoint_name
                #
                # torch.save(self.model.module.gnn.state_dict(), checkpoint_name)
                # print("+++model saved ! {}.{}.{}.pth".format(dataname, self.pretext, self.gnn_type))

            # Perform testing every 5 epochs
            if epoch % 5 == 0:
                test_metrics = self.test_model(self.model, test_loader)
                print("****Test Results at epoch {}: {}".format(epoch, test_metrics))


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

    # mkdir('./pre_trained_gnn/')
    # do not use '../pre_trained_gnn/' because hope there should be two folders: (1) '../pre_trained_gnn/'  and (2) './pre_trained_gnn/'
    # only selected pre-trained models will be moved into (1) so that we can keep reproduction

    pretext = 'GraphCL'
    # pretext = 'SimGRACE'
    gnn_type = 'TransformerConv'  
    # gnn_type = 'GAT'
    # gnn_type = 'GCN'
    dataname, num_parts = 'Prostate', 200
    graph_list, input_dim, hid_dim = load_data4pretrain(dataname, num_parts)

    pt = Downstream(pretext, gnn_type, input_dim, hid_dim, gln=2, combine_mode='graph_level', post_mode='abmil')
    pt.model.to(device) 
    pt.train(dataname, graph_list, batch_size=10, aug1='dropN', aug2="permE", aug_ratio=None,lr=0.01, decay=0.0001,epochs=100)
