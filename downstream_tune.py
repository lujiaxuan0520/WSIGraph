import argparse
from ProG.utils import seed_everything, seed
# from torchsummary import summary
from ProG.utils import print_model_parameters
from ProG.prompt import HeavyPrompt
from ProG.utils import center_embedding, Gprompt_tuning_loss, constraint, load_data4pretrain
# from ProG.evaluation import GpromptEva, GNNGraphEva, GPFEva, AllInOneEva, GPPTGraphEva
# from ProG.Prompt import GPF, GPF_plus, LightPrompt,HeavyPrompt, Gprompt, GPPTPrompt, DiffPoolPrompt, SAGPoolPrompt
# from ProG.prompt import GNN, ClusterGNN, SoftClusterGNN
from ProG.downstream import Downstream

import os
import numpy as np
import time
import torch
from torch import optim
from torch_geometric.loader import DataLoader


dataset_clss_num ={'Prostate': 4, 'JinYu': 21}

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GraphCL', type=str) # 'GraphCL', 'SimGRACE'

    parser.add_argument('--gnn', default='TransformerConv', type=str) # 'GAT', 'GCN'
    parser.add_argument('--mode', default='hard', type=str)  # 'original', 'hard', 'soft'
    parser.add_argument('--combine_mode', default='hier_mean', type=str)  # 'graph_level', 'region_level', 'hier_mean', 'hier_weighted_mean'
    parser.add_argument('--post_mode', default=None, type=str)  # None, 'linear_probing', 'abmil'
    parser.add_argument('--layer', default=2, type=int) # the layer of GNN
    parser.add_argument('--cluster_sizes', nargs='+', type=int, help='cluster size for each layer') # 100,50,10
    parser.add_argument('--dataset', default='CiteSeer', type=str) # 'CiteSeer', 'prostate'
    parser.add_argument('--encoder', default=None, type=str)  # 'Pathoduet', 'ResNet50', 'None'
    parser.add_argument('--encoder_path', default=None, type=str)
    parser.add_argument('--checkpoint_suffix', default='', type=str) # the suffix of checkpoint name
    parser.add_argument('--loss', default='MultiLabelBCELoss', type=str)  # 'WeightedCrossEntropyLoss', 'MultiLabelBCELoss'
    parser.add_argument('--gnn_ckpt', default='', type=str) # the pretrained gnn checkpoint for finetuning

    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--num_parts', default=200, type=int)
    parser.add_argument('--num_workers', default=1, type=int)

    args = parser.parse_args()

    return args


args = make_parse()


# class BaseTask:
#     def __init__(self, pre_train_model_path=None, gnn_type='TransformerConv', hid_dim=128, num_layer=2,
#                  dataset_name='Cora', prompt_type='GPF', epochs=100, shot_num=10, device: int = 5):
#         self.pre_train_model_path = pre_train_model_path
#         self.device = torch.device('cuda:' + str(device) if torch.cuda.is_available() else 'cpu')
#         self.hid_dim = hid_dim
#         self.num_layer = num_layer
#         self.dataset_name = dataset_name
#         self.shot_num = shot_num
#         self.gnn_type = gnn_type
#         self.prompt_type = prompt_type
#         self.epochs = epochs
#         self.initialize_lossfn()
#
#     def initialize_optimizer(self):
#         if self.prompt_type == 'None':
#             model_param_group = []
#             model_param_group.append({"params": self.gnn.parameters()})
#             model_param_group.append({"params": self.answering.parameters()})
#             self.optimizer = optim.Adam(model_param_group, lr=0.005, weight_decay=5e-4)
#         elif self.prompt_type == 'All-in-one':
#             self.pg_opi = optim.Adam(filter(lambda p: p.requires_grad, self.prompt.parameters()), lr=0.001,
#                                      weight_decay=0.00001)
#             self.answer_opi = optim.Adam(filter(lambda p: p.requires_grad, self.answering.parameters()), lr=0.001,
#                                          weight_decay=0.00001)
#         elif self.prompt_type in ['GPF', 'GPF-plus']:
#             model_param_group = []
#             model_param_group.append({"params": self.prompt.parameters()})
#             model_param_group.append({"params": self.answering.parameters()})
#             self.optimizer = optim.Adam(model_param_group, lr=0.005, weight_decay=5e-4)
#         elif self.prompt_type in ['Gprompt', 'GPPT']:
#             self.pg_opi = optim.Adam(self.prompt.parameters(), lr=0.01, weight_decay=5e-4)
#         elif self.prompt_type == 'MultiGprompt':
#             self.optimizer = torch.optim.Adam([*self.DownPrompt.parameters(), *self.feature_prompt.parameters()],
#                                               lr=0.001)
#
#     def initialize_lossfn(self):
#         self.criterion = torch.nn.CrossEntropyLoss()
#         if self.prompt_type == 'Gprompt':
#             self.criterion = Gprompt_tuning_loss()
#
#     def initialize_prompt(self):
#         if self.prompt_type == 'None':
#             self.prompt = None
#         elif self.prompt_type == 'GPPT':
#             if (self.task_type == 'NodeTask'):
#                 self.prompt = GPPTPrompt(self.hid_dim, self.output_dim, self.output_dim, device=self.device)
#                 train_ids = torch.nonzero(self.data.train_mask, as_tuple=False).squeeze()
#                 node_embedding = self.gnn(self.data.x, self.data.edge_index)
#                 self.prompt.weigth_init(node_embedding, self.data.edge_index, self.data.y, train_ids)
#             elif (self.task_type == 'GraphTask'):
#                 self.prompt = GPPTPrompt(self.hid_dim, self.output_dim, self.output_dim, device=self.device)
#         elif self.prompt_type == 'All-in-one':
#             lr, wd = 0.001, 0.00001
#             # self.prompt = LightPrompt(token_dim=self.input_dim, token_num_per_group=100, group_num=self.output_dim, inner_prune=0.01).to(self.device)
#             self.prompt = HeavyPrompt(token_dim=self.input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3).to(
#                 self.device)
#         elif self.prompt_type == 'GPF':
#             self.prompt = GPF(self.input_dim).to(self.device)
#         elif self.prompt_type == 'GPF-plus':
#             self.prompt = GPF_plus(self.input_dim, 20).to(self.device)
#         elif self.prompt_type == 'sagpool':
#             self.prompt = SAGPoolPrompt(self.input_dim, num_clusters=5, ratio=0.5).to(self.device)
#         elif self.prompt_type == 'diffpool':
#             self.prompt = DiffPoolPrompt(self.input_dim, num_clusters=5).to(self.device)
#         elif self.prompt_type == 'Gprompt':
#             self.prompt = Gprompt(self.hid_dim).to(self.device)
#         elif self.prompt_type == 'MultiGprompt':
#             nonlinearity = 'prelu'
#             self.Preprompt = PrePrompt(self.dataset_name, self.hid_dim, nonlinearity, 0.9, 0.9, 0.1, 0.001, 1, 0.3).to(
#                 self.device)
#             self.Preprompt.load_state_dict(torch.load(self.pre_train_model_path))
#             self.Preprompt.eval()
#             self.feature_prompt = featureprompt(self.Preprompt.dgiprompt.prompt,
#                                                 self.Preprompt.graphcledgeprompt.prompt,
#                                                 self.Preprompt.lpprompt.prompt).to(self.device)
#             dgiprompt = self.Preprompt.dgi.prompt
#             graphcledgeprompt = self.Preprompt.graphcledge.prompt
#             lpprompt = self.Preprompt.lp.prompt
#             self.DownPrompt = downprompt(dgiprompt, graphcledgeprompt, lpprompt, 0.001, self.hid_dim, 7,
#                                          self.device).to(self.device)
#         else:
#             raise KeyError(" We don't support this kind of prompt.")
#
#     def initialize_gnn(self):
#         if self.gnn_type == 'GAT':
#             self.gnn = GAT(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
#         elif self.gnn_type == 'GCN':
#             self.gnn = GCN(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
#         elif self.gnn_type == 'GraphSAGE':
#             self.gnn = GraphSAGE(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
#         elif self.gnn_type == 'GIN':
#             self.gnn = GIN(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
#         elif self.gnn_type == 'GCov':
#             self.gnn = GCov(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
#         elif self.gnn_type == 'GraphTransformer':
#             self.gnn = GraphTransformer(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
#         else:
#             raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
#         self.gnn.to(self.device)
#
#         if self.pre_train_model_path != 'None' and self.prompt_type != 'MultiGprompt':
#             if self.gnn_type not in self.pre_train_model_path:
#                 raise ValueError(f"the Downstream gnn '{self.gnn_type}' does not match the pre-train model")
#             if self.dataset_name not in self.pre_train_model_path:
#                 raise ValueError(f"the Downstream dataset '{self.dataset_name}' does not match the pre-train dataset")
#
#             self.gnn.load_state_dict(torch.load(self.pre_train_model_path, map_location=self.device))
#             print("Successfully loaded pre-trained weights!")
#
#
# class GraphTask(BaseTask):
#     def __init__(self, *argss, **kwargs):
#         super().__init__(*argss, **kwargs)
#         self.task_type = 'GraphTask'
#
#         self.load_data()
#         # self.create_few_data_folder()
#         self.initialize_gnn()
#         self.initialize_prompt()
#         self.answering = torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
#                                              torch.nn.Softmax(dim=1)).to(self.device)
#         self.initialize_optimizer()
#
#     def create_few_data_folder(self):
#         # 创建文件夹并保存数据
#         for k in range(1, 11):
#             k_shot_folder = './Experiment/sample_data/Graph/' + self.dataset_name + '/' + str(k) + '_shot'
#             os.makedirs(k_shot_folder, exist_ok=True)
#
#             for i in range(1, 6):
#                 folder = os.path.join(k_shot_folder, str(i))
#                 os.makedirs(folder, exist_ok=True)
#                 graph_sample_and_save(self.dataset, k, folder, self.output_dim)
#                 print(str(k) + ' shot ' + str(i) + ' th is saved!!')
#
#     def load_data(self):
#         if self.dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2',
#                                  'BZR', 'PTC_MR']:
#             self.input_dim, self.output_dim, self.dataset = load4graph(self.dataset_name, self.shot_num)
#
#     def Train(self, train_loader):
#         self.gnn.train()
#         total_loss = 0.0
#         for batch in train_loader:
#             self.optimizer.zero_grad()
#             batch = batch.to(self.device)
#             out = self.gnn(batch.x, batch.edge_index, batch.batch)
#             out = self.answering(out)
#             loss = self.criterion(out, batch.y)
#             loss.backward()
#             self.optimizer.step()
#             total_loss += loss.item()
#         return total_loss / len(train_loader)
#
#     def AllInOneTrain(self, train_loader, answer_epoch=1, prompt_epoch=1):
#         # we update answering and prompt alternately.
#
#         # answer_epoch = 1  # 50
#         # prompt_epoch = 1  # 50
#         # answer_epoch = 5  # 50  #PROTEINS # COX2
#         # prompt_epoch = 1  # 50
#
#         # tune task head
#         self.answering.train()
#         self.prompt.eval()
#         for epoch in range(1, answer_epoch + 1):
#             answer_loss = self.prompt.Tune(train_loader, self.gnn, self.answering, self.criterion, self.answer_opi,
#                                            self.device)
#             print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch,
#                                                                                                           answer_epoch,
#                                                                                                           answer_loss)))
#
#         # tune prompt
#         self.answering.eval()
#         self.prompt.train()
#         for epoch in range(1, prompt_epoch + 1):
#             pg_loss = self.prompt.Tune(train_loader, self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
#             print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch,
#                                                                                                          answer_epoch,
#                                                                                                          pg_loss)))
#
#         return pg_loss
#
#     def GPFTrain(self, train_loader):
#         self.prompt.train()
#         total_loss = 0.0
#         for batch in train_loader:
#             self.optimizer.zero_grad()
#             batch = batch.to(self.device)
#             batch.x = self.prompt.add(batch.x)
#             out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt=self.prompt, prompt_type=self.prompt_type)
#             out = self.answering(out)
#             loss = self.criterion(out, batch.y)
#             loss.backward()
#             self.optimizer.step()
#             total_loss += loss.item()
#         return total_loss / len(train_loader)
#
#     def GpromptTrain(self, train_loader):
#         self.prompt.train()
#         total_loss = 0.0
#         accumulated_centers = None
#         accumulated_counts = None
#
#         for batch in train_loader:
#
#             # archived code for complete prototype embeddings of each labels. Not as well as batch version
#             # # compute the prototype embeddings of each type of label
#
#             self.pg_opi.zero_grad()
#             batch = batch.to(self.device)
#             out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt=self.prompt, prompt_type='Gprompt')
#             # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
#             center, class_counts = center_embedding(out, batch.y, self.output_dim)
#             # 累积中心向量和样本数
#             if accumulated_centers is None:
#                 accumulated_centers = center
#                 accumulated_counts = class_counts
#             else:
#                 accumulated_centers += center * class_counts
#                 accumulated_counts += class_counts
#             criterion = Gprompt_tuning_loss()
#             loss = criterion(out, center, batch.y)
#             loss.backward()
#             self.pg_opi.step()
#             total_loss += loss.item()
#             # 计算加权平均中心向量
#             mean_centers = accumulated_centers / accumulated_counts
#
#             return total_loss / len(train_loader), mean_centers
#
#     def GPPTtrain(self, train_loader):
#         self.prompt.train()
#         for batch in train_loader:
#             temp_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
#             graph_list = batch.to_data_list()
#             for graph in graph_list:
#                 graph = graph.to(self.device)
#                 node_embedding = self.gnn(graph.x, graph.edge_index)
#                 out = self.prompt(node_embedding, graph.edge_index)
#                 loss = self.criterion(out,
#                                       torch.full((1, graph.x.shape[0]), graph.y.item()).reshape(-1).to(self.device))
#                 temp_loss += loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())
#             self.pg_opi.zero_grad()
#             temp_loss.backward()
#             self.pg_opi.step()
#             self.prompt.update_StructureToken_weight(self.prompt.get_mid_h())
#         return temp_loss.item()
#
#     def run(self):
#         test_accs = []
#         for i in range(1, 6):
#
#             idx_train = torch.load(
#                 "./Experiment/sample_data/Graph/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.shot_num,
#                                                                                    i)).type(torch.long).to(self.device)
#             print('idx_train', idx_train)
#             train_lbls = torch.load(
#                 "./Experiment/sample_data/Graph/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.shot_num,
#                                                                                       i)).type(torch.long).squeeze().to(
#                 self.device)
#             print("true", i, train_lbls)
#
#             idx_test = torch.load(
#                 "./Experiment/sample_data/Graph/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.shot_num,
#                                                                                   i)).type(torch.long).to(self.device)
#             test_lbls = torch.load(
#                 "./Experiment/sample_data/Graph/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.shot_num,
#                                                                                      i)).type(torch.long).squeeze().to(
#                 self.device)
#
#             train_dataset = self.dataset[idx_train]
#             test_dataset = self.dataset[idx_test]
#             train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#             test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#             print("prepare data is finished!")
#
#             patience = 20
#             best = 1e9
#             cnt_wait = 0
#
#             if self.prompt_type == 'All-in-one':
#                 self.answer_epoch = 5
#                 self.prompt_epoch = 1
#                 self.epochs = int(self.epochs / self.answer_epoch)
#             elif self.prompt_type == 'GPPT':
#                 # initialize the GPPT hyperparametes via graph data
#                 train_node_ids = torch.arange(0, train_dataset.x.shape[0]).squeeze().to(self.device)
#                 self.gppt_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
#                 for i, batch in enumerate(self.gppt_loader):
#                     if (i == 0):
#                         node_for_graph_labels = torch.full((1, batch.x.shape[0]), batch.y.item())
#                     else:
#                         node_for_graph_labels = torch.concat(
#                             [node_for_graph_labels, torch.full((1, batch.x.shape[0]), batch.y.item())], dim=1)
#                 node_embedding = self.gnn(self.dataset.x.to(self.device), self.dataset.edge_index.to(self.device))
#                 node_for_graph_labels = node_for_graph_labels.reshape((-1)).to(self.device)
#                 self.prompt.weigth_init(node_embedding, self.dataset.edge_index.to(self.device), node_for_graph_labels,
#                                         train_node_ids)
#
#                 test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#
#                 # from torch_geometric.nn import global_mean_pool
#                 # self.gppt_pool = global_mean_pool
#                 # train_ids = torch.nonzero(idx_train, as_tuple=False).squeeze()
#                 # self.gppt_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)          
#                 # for i, batch in enumerate(self.gppt_loader):
#                 #     batch.to(self.device)
#                 #     node_embedding = self.gnn(batch.x, batch.edge_index)
#                 #     if(i==0):
#                 #         graph_embedding = self.gppt_pool(node_embedding,batch.batch.long())
#                 #     else:
#                 #         graph_embedding = torch.concat([graph_embedding,self.gppt_pool(node_embedding,batch.batch.long())],dim=0)
#
#             for epoch in range(1, self.epochs + 1):
#                 t0 = time.time()
#
#                 if self.prompt_type == 'None':
#                     loss = self.Train(train_loader)
#                 elif self.prompt_type == 'All-in-one':
#                     loss = self.AllInOneTrain(train_loader, self.answer_epoch, self.prompt_epoch)
#                 elif self.prompt_type in ['GPF', 'GPF-plus']:
#                     loss = self.GPFTrain(train_loader)
#                 elif self.prompt_type == 'Gprompt':
#                     loss, center = self.GpromptTrain(train_loader)
#                 elif self.prompt_type == 'GPPT':
#                     loss = self.GPPTtrain(train_loader)
#
#                 if loss < best:
#                     best = loss
#                     # best_t = epoch
#                     cnt_wait = 0
#                     # torch.save(model.state_dict(), args.save_name)
#                 else:
#                     cnt_wait += 1
#                     if cnt_wait == patience:
#                         print('-' * 100)
#                         print('Early stopping at ' + str(epoch) + ' eopch!')
#                         break
#                 print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f}  ".format(epoch, time.time() - t0, loss))
#
#             if self.prompt_type == 'None':
#                 test_acc = GNNGraphEva(test_loader, self.gnn, self.answering, self.device)
#             elif self.prompt_type == 'All-in-one':
#                 test_acc, F1 = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim,
#                                            self.device)
#             elif self.prompt_type in ['GPF', 'GPF-plus']:
#                 test_acc = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.device)
#             elif self.prompt_type == 'Gprompt':
#                 test_acc = GpromptEva(test_loader, self.gnn, self.prompt, center, self.device)
#             elif self.prompt_type == 'GPPT':
#                 test_acc = GPPTGraphEva(test_loader, self.gnn, self.prompt, self.device)
#
#             print("test accuracy {:.4f} ".format(test_acc))
#             test_accs.append(test_acc)
#
#         mean_test_acc = np.mean(test_accs)
#         std_test_acc = np.std(test_accs)
#         print(" Final best | test Accuracy {:.4f} | std {:.4f} ".format(mean_test_acc, std_test_acc))
#
#         print("Graph Task completed")


if __name__ == '__main__':

    seed_everything(seed)
    # args.task = 'GraphTask'
    # args.prompt_type = 'Gprompt'
    args.shot_num = 10
    # args.dataset_name = 'PROTEINS'
    # args.task = 'NodeTask'
    args.epochs = 10
    # args.dataset_name = 'CiteSeer'

    args.num_layer = 2

    args.prompt_type = 'MultiGprompt'

    args.pre_train_model_path = './pre_trained_gnn/Prostate.GraphCL.GCN.GCN_soft_pool_cluster_200_100_50_SGD_lr_0.0001_batch_3_worker_32_epoch_100_loss_0.4705.pth'

    pretext = args.model
    gnn_type = args.gnn
    encoder = args.encoder
    encoder_path = args.encoder_path
    dataname, num_parts, batch_size = args.dataset, args.num_parts, args.batch_size

    print("load data...")
    train_graph_list = load_data4pretrain(args.dataset, args.num_parts, phase='finetune')
    test_graph_list = load_data4pretrain(args.dataset, args.num_parts, phase='test')

    print("create Downstream instance...")
    pt = Downstream(pretext, gnn_type, encoder, encoder_path, gln=args.layer, cluster_sizes=args.cluster_sizes,
                    mode=args.mode, post_mode=args.post_mode, num_workers=args.num_workers, combine_mode=args.combine_mode,
                    class_num=dataset_clss_num[dataname], loss_name=args.loss, gnn_ckpt=args.gnn_ckpt)

    print("fine-tuning...")
    pt.train(dataname, train_graph_list, test_graph_list, batch_size=batch_size, lr=args.learning_rate, decay=0.0001, epochs=100,
             checkpoint_suffix=args.checkpoint_suffix, save_epoch=True)


    # tasker = GraphTask(pre_train_model_path=args.pre_train_model_path,
    #                    dataset_name=args.dataset, num_layer=args.num_layer, gnn_type=args.gnn,
    #                    prompt_type=args.prompt_type, epochs=args.epochs, shot_num=args.shot_num, device="0")
    # tasker.run()