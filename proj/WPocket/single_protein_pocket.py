from wcode.model.GINEConv import GraphEmbeddingModel
from wcode.protein.graph.graph import construct_graph
from wcode.protein.graph.graph_conversion import GraphFormatConvertor
from torch.nn import Linear, Module, Sigmoid
from torch_geometric.data import Batch
from wcode.protein.convert import ProtConvertor
import torch
import os


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

    def forward(self, pyg):
        g = Batch.from_data_list([pyg])
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
    pyg, df, g = generate_pyg_feature_file("/mnt/c/tmp/8UAQ.pdb")
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
    ProtConvertor.df2pdb(raw_df, "/mnt/c/tmp/8UAQ_pred.pdb")