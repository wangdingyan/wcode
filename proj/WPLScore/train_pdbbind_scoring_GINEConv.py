import os
import torch
from glob import glob
from wcode.model.GINEConv import GraphEmbeddingModel
from torch_geometric.data import Batch
from torch.nn import Linear, Module
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from scipy.stats import pearsonr
from tqdm import tqdm

label_dict = {}
with open('C:\\database\\PDBBind\\INDEX_general_PL_data.2020', 'r') as f:
    lines = f.readlines()[6:]
    for line in lines:
        l = line.split()
        label_dict[l[0]] = float(l[3])

test_names = os.listdir('C:\\database\\PDBBind\\coreset')
complete_names = glob('C:\\database\\PDBBind\\PDBBind_pyg_feature\\*.pt')
complete_names = [os.path.basename(n).split('_')[0] for n in complete_names]

valid_test_name = [n for n in test_names if os.path.exists(f'C:\\database\\PDBBind\\PDBBind_pyg_feature\\{n}_pyg.pt')]
valid_train_names = [n for n in complete_names if n not in valid_test_name]

class scoring_model(Module):
    def __init__(self):
        super().__init__()
        self.embedding = GraphEmbeddingModel(142,
                                             5,
                                             0,
                                             128,
                                             100)
        self.output_layer = Linear(100, 1)

    def forward(self, list_of_pdb_id):
        g = Batch.from_data_list([torch.load(f'C:\\database\\PDBBind\\PDBBind_pyg_feature\\{n}_pyg.pt')
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
                                   node2graph=g.batch.cuda())[1]
        output = self.output_layer(embedding).squeeze()
        return output

model = scoring_model()
model = model.cuda()
train_dataloader = DataLoader(valid_train_names, batch_size=50, shuffle=True)
test_dataloader = DataLoader(valid_test_name, batch_size=50, shuffle=False)
loss = MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)
for epoch in range(1000):
    print(f'Eval at {epoch} Epoch:')
    with torch.no_grad():
        model.eval()
        test_labels = []
        test_preds = []
        for batch in tqdm(test_dataloader):
            prediction = model(batch).cpu().tolist()
            labels = [label_dict[s] for s in batch]
            test_labels.extend(labels)
            test_preds.extend(prediction)
        print(f'pearsonr: {pearsonr(test_labels, test_preds)[0]}')

    model.train()
    for batch in tqdm(train_dataloader):
        try:
            prediction = model(batch)
            labels = torch.Tensor([label_dict[s] for s in batch]).cuda()
            optimizer.zero_grad()
            l = loss(prediction, labels)
            l.backward()
            optimizer.step()
        except:
            print("Error. Skip.")










