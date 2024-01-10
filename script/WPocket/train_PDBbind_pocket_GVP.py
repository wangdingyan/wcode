from glob import glob
from wcode.model.GVP import GVPConvLayer
import torch.nn as nn
from torch.nn import Module, Sigmoid, BCELoss
from torch_geometric.data import Batch
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from torch.optim import Adam
from tqdm import tqdm

class pocket_dataset():
    def __init__(self, mode):
        paths = glob('/mnt/c/database/PDBBind/pocket_marked_CA_esm_dssp/*_dssp.pt')
        length = len(paths)
        if mode == 'train':
            self.paths = paths[:int(length * 0.9)]
        elif mode == 'valid':
            self.paths = paths[int(length * 0.9):]

    def __getitem__(self, i):
        return self.paths[i]

    def __len__(self):
        return len(self.paths)


class scoring_model(Module):
    def __init__(self):
        super().__init__()
        self.linear_scalar = nn.Linear(1433, 128)

        self.embedding1 = GVPConvLayer((128, 2),
                                      (8,1))
        self.embedding2 = GVPConvLayer((128, 2),
                                      (8,1))

        self.read_out = nn.Linear(128, 1)
        self.activation = Sigmoid()

    def forward(self, list_of_pdb_id):
        g = Batch.from_data_list([torch.load(n)
                                  for n in list_of_pdb_id])

        atom_scalar_feature = torch.concat([g.residue_name_one_hot, # 23
                                                   g.atom_type_one_hot, # 38
                                                   g.record_symbol_one_hot, # 3
                                                   g.rdkit_atom_feature_onehot, # 78
                                                   g.esm_embedding, # 1280
                                                   g.ss_onehot, # 8
                                                   g.phi, # 1
                                                   g.psi, # 1
                                                   g.rsa  # 1
                                                     ], dim=-1).float().cuda()
        atom_scalar_feature = self.linear_scalar(atom_scalar_feature)
        atom_vector_feature = torch.concat([g.ToNextCA.reshape(-1,1,3),
                                                   g.ToLastCA.reshape(-1,1,3)], dim=-2).float().cuda()

        edge_index = torch.Tensor(g.edge_index).long().cuda()
        bond_scalar_feature = torch.Tensor(g.distance_fourier).float().cuda()
        bond_vector_feature = torch.Tensor(g.direction_vector).reshape(-1,1,3).float().cuda()

        embedding = self.embedding1(tuple((atom_scalar_feature, atom_vector_feature)),
                                   edge_index,
                                   tuple((bond_scalar_feature, bond_vector_feature)))
        embedding = self.embedding2(tuple((embedding[0], embedding[1])),
                                   edge_index,
                                   tuple((bond_scalar_feature, bond_vector_feature)))[0]


        labels = g.b_factor.cuda().squeeze()
        output = self.activation(self.read_out(embedding).squeeze())

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
        torch.save(model.cpu().state_dict(), "/mnt/c/tmp/pocket_prediction_model_20240108.pt")
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
        try:

            model.train()
            o, l = model(k)
            l = l.float()
            loss = loss_fn(o, l)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            o_numpy = o.cpu().detach().numpy()
            l_numpy = l.cpu().detach().numpy()
        except:
            pass
