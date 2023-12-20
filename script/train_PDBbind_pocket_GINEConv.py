from glob import glob
from wcode.model.GINEConv import GraphEmbeddingModel
from torch.nn import Linear, Module, Sigmoid, BCELoss
from torch_geometric.data import Batch
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from torch.optim import Adam

class pocket_dataset():
    def __init__(self):
        self.paths = glob('D:\\PDBBind\\PDBBind_processed\\*\\*_marked.pt')

    def __getitem__(self, i):
        return self.paths[i]

    def __len__(self):
        return len(self.paths)


class scoring_model(Module):
    def __init__(self):
        super().__init__()
        self.embedding = GraphEmbeddingModel(142,
                                             5,
                                             0,
                                             128,
                                             0)
        self.output_layer = Linear(128, 1)
        self.activation = Sigmoid()

    def forward(self, list_of_pdb_id):
        g = Batch.from_data_list([torch.load(n)
                                  for n in list_of_pdb_id])
        atom_feature = torch.concat([g.residue_name_one_hot,
                                     g.atom_type_one_hot,
                                     g.record_symbol_one_hot,
                                     g.rdkit_atom_feature_onehot], dim=-1).float().cuda()
        edge_index = torch.Tensor(g.edge_index).long().cuda()
        bond_feature = torch.Tensor(g.bond_feature).float().cuda()
        embedding = self.embedding(atom_feature,
                                   edge_index,
                                   bond_feature,
                                   node2graph=g.batch.cuda())[0]
        labels = g.b_factor.cuda()
        output = self.activation(self.output_layer(embedding).squeeze())
        return output, labels


model = scoring_model()
model = model.cuda()
dataset = pocket_dataset()
train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = BCELoss()

for e in range(100):
    for k in train_dataloader:
        model.train()
        o, l = model(k)
        l = l.float()
        loss = loss_fn(o, l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        o_numpy = o.cpu().detach().numpy()
        l_numpy = l.cpu().detach().numpy()
        print(round(loss.cpu().detach().item(), 3),metrics.roc_auc_score(l_numpy, o_numpy))