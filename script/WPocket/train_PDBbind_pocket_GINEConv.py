from glob import glob
from wcode.model.GINEConv import GraphEmbeddingModel
from torch.nn import Linear, Module, Sigmoid, BCELoss
from torch_geometric.data import Batch
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from torch.optim import Adam
from tqdm import tqdm
from wcode.model.EGNN import fourier_encode_dist

class pocket_dataset():
    def __init__(self, mode):
        paths = glob('/mnt/c/database/PDBBind/pocket_marked_CA_esm_dssp/*_dssp.pt')
        length = len(paths)
        if mode == 'train':
            self.paths = paths[:int(length * 0.8)]
        elif mode == 'valid':
            self.paths = paths[int(length * 0.8):]

    def __getitem__(self, i):
        return self.paths[i]

    def __len__(self):
        return len(self.paths)


class scoring_model(Module):
    def __init__(self):
        super().__init__()
        self.embedding = GraphEmbeddingModel(1422,
                                             8,
                                             0,
                                             128,
                                             0,
                                             n_block=5,
                                             dropout=0.2)
        self.output_layer = Linear(128, 1)
        self.activation = Sigmoid()

    def forward(self, list_of_pdb_id):
        g = Batch.from_data_list([torch.load(n)
                                  for n in list_of_pdb_id])
        atom_feature = torch.concat([g.residue_name_one_hot,
                                     g.atom_type_one_hot,
                                     g.record_symbol_one_hot,
                                     g.rdkit_atom_feature_onehot,
                                     g.esm_embedding], dim=-1).float().cuda()
        edge_index = torch.Tensor(g.edge_index).long().cuda()
        bond_feature = torch.Tensor(g.distance_fourier).float().cuda()

        embedding = self.embedding(atom_feature,
                                   edge_index,
                                   bond_feature,
                                   node2graph=g.batch.cuda())[0]
        labels = g.b_factor.cuda().squeeze()
        output = self.activation(self.output_layer(embedding).squeeze())
        return output, labels


model = scoring_model()
model = model.cuda()
train_dataset = pocket_dataset(mode='train')
valid_dataset = pocket_dataset(mode='valid')
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = BCELoss()

for e in range(200):
    with torch.no_grad():
        torch.save(model.cpu().state_dict(), "/mnt/c/tmp/pocket_prediction_model.pt")
        model.cuda()
        model.eval()
        test_labels = []
        test_preds = []
        for batch in tqdm(valid_dataloader):
            o, l = model(batch)
            l = l.detach().cpu().tolist()
            o = o.detach().cpu().tolist()

            test_labels.extend(l)
            test_preds.extend(o)
        print(f'auROC: {metrics.roc_auc_score(test_labels, test_preds)}')
    for k in tqdm(train_dataloader):

        model.train()
        o, l = model(k)
        l = l.float()
        loss = loss_fn(o, l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        o_numpy = o.cpu().detach().numpy()
        l_numpy = l.cpu().detach().numpy()
