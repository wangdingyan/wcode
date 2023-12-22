from wcode.model.GINEConv import GraphEmbeddingModel
from torch.nn import Linear, Module, Sigmoid, BCELoss
from torch_geometric.data import Batch
import torch
from wcode.protein.convert import ProtConvertor


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

    def forward(self, g):
        # g = Batch.from_data_list([torch.load(n)
        #                           for n in list_of_pdb_id])
        g = Batch.from_data_list([g])
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
model.load_state_dict(torch.load("C:\\tmp\\pocket_prediction_model.pt"))
model = model.cuda()

converter = ProtConvertor()
nxg = converter.pdb2nx("C:\\tmp\\SLC6A2_Vilazodone_1550_coot-5.pdb", pocket_only=False)[0]
pyg = converter.nx2pyg(nxg)
pyg = pyg.cuda()
prediction = model(pyg)
pyg['b_factor'] = prediction[0]
pyg = pyg.cpu().detach()
nx = converter.pyg2nx(pyg)
df = converter.nx2df(nx)
converter.df2pdb(df, "C:\\tmp\\NET_prediction.pdb")