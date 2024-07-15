import os
import torch
from glob import glob
from torch.nn.utils.rnn import pad_sequence
from wcode.model.GINEConv import GraphEmbeddingModel
from wcode.model.EGNN import EGNN_Network
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
        self.embedding = EGNN_Network(
                        num_tokens=None,
                        num_positions=None,
                        dim=142,
                        depth=1,
                        num_nearest_neighbors=10,
                        coor_weights_clamp_value=2.,
                        fourier_features=10
                        # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
                    )
        self.output_layer = Linear(142, 1)

    def forward(self, list_of_pdb_id):
        g_batch = [torch.load(f'C:\\database\\PDBBind\\PDBBind_pyg_feature\\{n}_pyg.pt')
                                  for n in list_of_pdb_id]
        node_feature_batch = []
        coords_batch = []
        mask_batch = []
        for g in g_batch:
            g_batch_atom_feature = torch.concat([g.residue_name_one_hot,
                                                 g.atom_type_one_hot,
                                                 g.record_symbol_one_hot,
                                                 g.rdkit_atom_feature_onehot], dim=-1).float()
            g_mask = torch.ones(len(g_batch_atom_feature))
            node_feature_batch.append(g_batch_atom_feature)
            mask_batch.append(g_mask)
            coords_batch.append(g.coords)

        node_feature_batch = pad_sequence(node_feature_batch, batch_first=True).float().cuda()
        mask_batch = pad_sequence(mask_batch, batch_first=True).bool().cuda()
        coords_batch = pad_sequence(coords_batch, batch_first=True).float().cuda()

        embedding = self.embedding(node_feature_batch, coords_batch, mask=mask_batch)[0]
        graph_embedding_batch = [e[b].mean(dim=0) for e, b in zip(embedding, mask_batch)]
        embedding = torch.stack(graph_embedding_batch)

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










