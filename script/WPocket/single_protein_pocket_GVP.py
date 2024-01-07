from wcode.protein.graph.graph import construct_graph
from wcode.protein.graph.graph_conversion import GraphFormatConvertor
from wcode.protein.convert import ProtConvertor
from wcode.model.GVP import GVPConvLayer
import torch.nn as nn
from torch.nn import Linear, Module, Sigmoid, BCELoss
from torch_geometric.data import Batch
import torch


class scoring_model(Module):
    def __init__(self):
        super().__init__()
        self.linear_scalar = nn.Linear(1433, 128)

        self.embedding = GVPConvLayer((128, 2),
                                      (8,1))

        self.read_out = nn.Linear(128, 1)
        self.activation = Sigmoid()

    def forward(self, list_of_pdb_id):
        g = Batch.from_data_list([pyg])

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

        embedding = self.embedding(tuple((atom_scalar_feature, atom_vector_feature)),
                                   edge_index,
                                   tuple((bond_scalar_feature, bond_vector_feature)))[0]

        labels = g.b_factor.cuda().squeeze()
        output = self.activation(self.read_out(embedding).squeeze())

        return output, labels

def generate_pyg_feature_file(pdb_file):
    g, df = construct_graph(pdb_file,
                            granularity='CA',
                            dssp=True,
                            esm=True,
                            pocket_only=False)
    converter = GraphFormatConvertor()
    pyg = converter.convert_nx_to_pyg(g)

    return pyg, df, g


if __name__ == '__main__':
    model = scoring_model()
    model.load_state_dict(torch.load("/mnt/c/tmp/pocket_prediction_model.pt"))
    model = model.cuda()
    model.eval()
    pyg, df, g = generate_pyg_feature_file("/mnt/c/tmp/8DZT_G4P_protein.pdb")
    output = model.forward(pyg)[0]
    residue_to_bf_dict = {residue_id: b_factor.cpu().detach().item() for
                          residue_id, b_factor in zip(pyg["node_id"], output)}

    def update_node_id(row):
        residue_id = row["residue_id"]
        if residue_id in residue_to_bf_dict:
            b_factor = residue_to_bf_dict[residue_id]
        else:
            b_factor = 0
        return b_factor

    raw_df = g.graph["raw_pdb_df"]
    raw_df["residue_id"] = (
        raw_df["chain_id"].apply(str)
        + ":"
        + raw_df["residue_name"]
        + ":"
        + raw_df["residue_number"].apply(str)
    )


    raw_df['b_factor'] = raw_df.apply(update_node_id, axis=1)
    ProtConvertor.df2pdb(raw_df, "/mnt/c/tmp/NET_prediction_2.pdb")