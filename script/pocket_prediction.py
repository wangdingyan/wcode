import glob
import os.path

from wcode.model.GINEConv import GraphEmbeddingModel
from torch.nn import Linear, Module, Sigmoid, BCELoss
from torch_geometric.data import Batch
import torch
from wcode.protein.convert import ProtConvertor
from wcode.model.EGNN import fourier_encode_dist


class scoring_model(Module):
    def __init__(self):
        super().__init__()
        self.embedding = GraphEmbeddingModel(142,
                                             25,
                                             0,
                                             128,
                                             0,
                                             n_block=5,
                                             dropout=0.2)
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
        distance_feature = fourier_encode_dist(g.distance, num_encodings=10, include_self=False).squeeze().float().cuda()
        bond_feature = torch.cat([bond_feature, distance_feature], dim=-1)
        embedding = self.embedding(atom_feature,
                                   edge_index,
                                   bond_feature,
                                   node2graph=g.batch.cuda())[0]
        labels = g.b_factor.cuda()
        output = self.activation(self.output_layer(embedding).squeeze())
        return output, labels


model = scoring_model()
model.load_state_dict(torch.load("/mnt/c/tmp/pocket_prediction_model_20231227_2.pt"))
model = model.cuda()

protein_names = glob.glob('/mnt/c/dataset/posebusters_benchmark_set/*/*.pdb')
protein_names = [os.path.basename(n) for n in protein_names]
finished_names = os.listdir('/mnt/c/dataset/posebusters_pocket_prediction_2')
still_names = [n for n in protein_names if n not in finished_names]
protein_names = [f'/mnt/c/dataset/posebusters_benchmark_set/{n.replace(".pdb", "").replace("_protein", "")}/{n}' for n in still_names]
for n in protein_names:
    print(n)
    try:
        base_name = os.path.basename(n)
        converter = ProtConvertor()
        nxg = converter.pdb2nx(n, pocket_only=False)[0]
        pyg = converter.nx2pyg(nxg)
        pyg = pyg.cuda()
        prediction = model(pyg)
        print(prediction)
        pyg['b_factor'] = prediction[0]*100
        pyg = pyg.cpu().detach()
        nx = converter.pyg2nx(pyg)
        df = converter.nx2df(nx)
        residue_bfactor = df.groupby('residue_number')['b_factor'].max()
        valid_residue = residue_bfactor[residue_bfactor > 50]
        df_out = df[df['residue_number'].isin(valid_residue.index)]
        if len(df_out) == 0:
            valid_residue = residue_bfactor.sort_values()[::-1][:10]
            df_out = df[df['residue_number'].isin(valid_residue.index)]
        converter.df2pdb(df_out, f"/mnt/c/dataset/posebusters_pocket_prediction_2/{base_name}")
    except:
        continue