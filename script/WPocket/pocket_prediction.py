import glob
import os.path


from wcode.protein.graph.graph import construct_graph
from wcode.protein.graph.graph_conversion import GraphFormatConvertor
from wcode.protein.convert import ProtConvertor
from wcode.model.GVP import GVPConvLayer
import torch.nn as nn
from torch.nn import Module, Sigmoid
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

model = scoring_model()
model.load_state_dict(torch.load("/mnt/c/tmp/pocket_prediction_model_20240108.pt"))
model = model.cuda()
model.eval()

protein_names = glob.glob('/mnt/c/dataset/posebusters_benchmark_set/*/*.pdb')
protein_names = [os.path.basename(n) for n in protein_names]
finished_names = os.listdir('/mnt/c/dataset/posebusters_pocket_prediction_3')
still_names = [n for n in protein_names if n not in finished_names]
failed_names = ['6YQV_8K2_protein.pdb',
                '7B2C_TP7_protein.pdb',
                '7BHX_TO5_protein.pdb',
                '7D0P_1VU_protein.pdb',
                '7D6O_MTE_protein.pdb',
                '7L00_XCJ_protein.pdb',
                '7L03_F9F_protein.pdb',
                '7LT0_ONJ_protein.pdb',
                '7LZD_YHY_protein.pdb',
                '7N7B_T3F_protein.pdb',
                '7N7H_CTP_protein.pdb',
                '7P1M_4IU_protein.pdb',
                '7P2I_MFU_protein.pdb',
                '7PIH_7QW_protein.pdb',
                '7POM_7VZ_protein.pdb',
                '7QE4_NGA_protein.pdb',
                '7RZL_NPO_protein.pdb',
                '7SUC_COM_protein.pdb',
                '7W05_GMP_protein.pdb',
                '7WUY_76N_protein.pdb',
                '7X5N_5M5_protein.pdb',
                '7ZXV_45D_protein.pdb',
                '8AIE_M7L_protein.pdb']
still_names = [n for n in still_names if n not in failed_names]
protein_names = [f'/mnt/c/dataset/posebusters_benchmark_set/{n.replace(".pdb", "").replace("_protein", "")}/{n}' for n in still_names]
for n in protein_names:
    print(n)

    # base_name = os.path.basename(n)
    # converter = ProtConvertor()
    # nxg = converter.pdb2nx(n, pocket_only=False)[0]
    # pyg = converter.nx2pyg(nxg)
    # pyg = pyg.cuda()
    # prediction = model(pyg)
    # print(prediction)
    # pyg['b_factor'] = prediction[0]*100
    # pyg = pyg.cpu().detach()
    # nx = converter.pyg2nx(pyg)
    # df = converter.nx2df(nx)
    # residue_bfactor = df.groupby('residue_number')['b_factor'].max()
    # valid_residue = residue_bfactor[residue_bfactor > 50]
    # df_out = df[df['residue_number'].isin(valid_residue.index)]
    # if len(df_out) == 0:
    #     valid_residue = residue_bfactor.sort_values()[::-1][:10]
    #     df_out = df[df['residue_number'].isin(valid_residue.index)]
    # converter.df2pdb(df_out, f"/mnt/c/dataset/posebusters_pocket_prediction_2/{base_name}")
    pyg, df, g = generate_pyg_feature_file(n)
    base_name = os.path.basename(n)
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
    raw_df = raw_df[raw_df['b_factor'] > 0.3]
    ProtConvertor.df2pdb(raw_df, f"/mnt/c/dataset/posebusters_pocket_prediction_3/{base_name}")