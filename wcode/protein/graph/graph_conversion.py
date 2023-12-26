import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.transforms import ToUndirected
from wcode.protein.graph.graph import construct_graph
from wcode.model.EGNN import fourier_encode_dist


########################################################################################################################

class GraphFormatConvertor:
    def __init__(self):
        columns = [
            # node feature
            "coords",
            "node_id",
            "residue_number",
            "element_symbol",
            "residue_name_one_hot",
            "atom_type_one_hot",
            "record_name",
            "record_symbol_one_hot",
            "rdkit_atom_feature",
            "rdkit_atom_feature_onehot",
            "node_id",
            "b_factor",

            # edge feature
            "edge_index",
            "distance",
            "bond_feature",

            # edge index
            "edge_index_covalent",
            "edge_index_noncovalent",
        ]
        self.columns = columns
        self.type2form = {
            "atom_type": "str",
            "b_factor": "float",
            "chain_id": "str",
            "coords": "np.array",
            "dist_mat": "np.array",
            "element_symbol": "str",
            "node_id": "str",
            "residue_name": "str",
            "residue_number": "int",
            "edge_index": "torch.tensor",
            "kind": "str",
        }

    def convert_nx_to_pyg(self, G):
        # Initialise dict used to construct Data object & Assign node ids as a feature
        data = {"node_id": list(G.nodes())}
        G = nx.convert_node_labels_to_integers(G)

        # Construct Edge Index
        edge_index = (
            torch.LongTensor(list(G.edges)).t().contiguous().view(2, -1)
        )

        # Add node features
        node_feature_names = G.nodes(data=True)[0].keys()

        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            for key, value in feat_dict.items():
                # if key == 'rdkit_atom_feature':
                #     with open('C:\\tmp\\check_dimension.txt', 'a') as f:
                #         f.write(str(value)+'\n')
                key = str(key)
                if key in self.columns:
                    if i == 0:
                        data[key] = []
                    data[key].append(value)

        # Add edge features
        edge_list = list(G.edges(data=True))
        key_lenth = {}

        for e in G.edges(data=True):
            e_attr = e[2]
            for k in e_attr.keys():
                if k not in key_lenth:
                    key_lenth[k] = len(e_attr[k])

        for e in G.edges(data=True):
            for k in key_lenth:
                if k not in e[2]:
                    e[2][k] = [-1.] * key_lenth[k]

        edge_feature_names = edge_list[0][2].keys() if edge_list else []

        edge_feature_names = list(
            filter(
                lambda x: x in self.columns and x != "kind", edge_feature_names
            )
        )
        for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
            for key, value in feat_dict.items():
                key = str(key)
                if key in self.columns or key == "kind":
                    if i == 0:
                        data[key] = []
                    data[key].append(value)

        # Add graph-level features
        for feat_name in G.graph:
            if str(feat_name) in self.columns:
                if str(feat_name) not in node_feature_names:
                    data[str(feat_name)] = G.graph[feat_name]

        if "edge_index" in self.columns:
            data["edge_index"] = edge_index

        # Split edge index by edge kind
        kind_strs = np.array(
            list(map(lambda x: "_".join(x), data.get("kind", [])))
        )

        for kind in set(kind_strs):
            key = f"edge_index_{kind}"
            if key in self.columns:
                mask = kind_strs == kind
                data[key] = edge_index[:, mask]
        if "kind" not in self.columns and data.get("kind"):
            del data["kind"]

        # Convert everything possible to torch.Tensors
        for key, val in data.items():
            try:
                if not isinstance(val, torch.Tensor):
                    data[key] = torch.tensor(np.array(val))
            except Exception as e:
                print(e)
                pass

        # Construct PyG data
        data = Data.from_dict(data)
        data.num_nodes = G.number_of_nodes()

        # Symmetrize if undirected
        if not G.is_directed():
            # Edge index and edge features
            edge_index, edge_features = to_undirected(
                edge_index,
                [getattr(data, attr) for attr in edge_feature_names],
                data.num_nodes,
            )
            if "edge_index" in self.columns:
                data.edge_index = edge_index
            for attr, val in zip(edge_feature_names, edge_features):
                setattr(data, attr, val)

            # Edge indices of different kinds
            for kind in set(kind_strs):
                key = f"edge_index_{kind}"
                if key in self.columns:
                    edge_index_kind = to_undirected(
                        getattr(data, key), num_nodes=data.num_nodes
                    )
                    setattr(data, key, edge_index_kind)

        return data

    def convert_to_hetero_graph(self, G):
        data = {}
        data['_global_store'] = {}
        data['_global_store']['initialnodeidx2hetero'] = {}

        for i, (node, feature) in enumerate(G.nodes(data=True)):
            node_type = get_node_type(node)
            if node_type not in data:
               data[node_type] = {}
               data[node_type]['node_idx'] = []
            hetero_length = len(data[node_type]['node_idx'])
            data[node_type]['node_idx'].append(hetero_length)
            data['_global_store']['initialnodeidx2hetero'][node] = (node_type, hetero_length)
            for k, v in feature.items():
                if k not in data[node_type]:
                    data[node_type][k] = []
                data[node_type][k].append(v)

        # Construct Edge Index
        for edge_start, edge_end, feature in G.edges(data=True):
            start_anchor = data['_global_store']['initialnodeidx2hetero'][edge_start]
            end_anchor = data['_global_store']['initialnodeidx2hetero'][edge_end]
            start_type = get_node_type(edge_start)
            end_type = get_node_type(edge_end)
            if (start_type, list(feature['kind'])[0], end_type) not in data:
                data[(start_type, list(feature['kind'])[0], end_type)] = {}
                data[(start_type, list(feature['kind'])[0], end_type)]['edge_index'] = [[],[]]
            data[(start_type, list(feature['kind'])[0], end_type)]['edge_index'][0].append(start_anchor[1])
            data[(start_type, list(feature['kind'])[0], end_type)]['edge_index'][1].append(end_anchor[1])

            for k, v in feature.items():
                if k == 'kind':
                    continue
                if k not in data[(start_type, list(feature['kind'])[0], end_type)]:
                    data[(start_type, list(feature['kind'])[0], end_type)][k] = []
                data[(start_type, list(feature['kind'])[0], end_type)][k].append(v)

        for k, v in data.items():
            if k == '_global_store':
                continue
            for k2, v2 in v.items():
                if type(v2[0]) == str:
                    continue
                v[k2] = torch.from_numpy(np.array(v2))
                if k2 == 'edge_index':
                    v[k2] = v[k2].long()

            if type(k) == str:
                data[k]['x'] = torch.cat([data[k]['residue_name_one_hot'],
                                          data[k]['atom_type_one_hot'],
                                          data[k]['element_symbol_one_hot'],
                                          data[k]['rdkit_atom_feature_onehot']], dim=-1)
            elif type(k) == tuple:

                data[k]['distance_fourier'] = fourier_encode_dist(data[k]['distance'], num_encodings=10, include_self=False).squeeze()
                if k[1] == 'noncovalent':
                    data[k]['edge_attr'] = data[k]['distance_fourier']
                elif k[1] == 'covalent':
                    data[k]['edge_attr'] = torch.cat([data[k]['distance_fourier'],
                                                             data[k]['bond_feature']], dim=-1)

        g = HeteroData.from_dict(data)
        g.num_nodes = G.number_of_nodes()
        transformer = ToUndirected(merge=False)
        g = transformer(g)
        return g




def get_node_type(node_id):
    if 'LIG' in node_id:
        return "ligand"
    elif "SUF" in node_id:
        return "surface"
    elif "RES" in node_id:
        return "residue"
    else:
        return "protein"


########################################################################################################################

# g, df = construct_graph('C:\\database\\PDBBind\\PDBBind_processed\\1a0t\\1a0t_protein_processed.pdb',
#                     'C:\\database\\PDBBind\\PDBBind_processed\\1a0t\\1a0t_ligand.sdf',
#                               pocket_only=True)
# converter = GraphFormatConvertor()
# G = converter.convert_nx_to_pyg(g)
#
# print(G)
# from torch import isclose
# print(isclose(G['coords'], G['rdkit_atom_feature'][:, -3:], atol=0.1).all())
def func(name):
    # if os.path.exists(f'C:\\database\\PDBBind\\PDBBind_pyg_feature\\{name}_pyg.pt'):
    #     print("HERE WE ARE", name)
    #     with open('C:\\tmp\\check.txt', 'a') as f:
    #         f.write(f'{name} pass\n')
    #     return

    # try:
    print(name)
    g, df = construct_graph(f'C:\\database\\PDBBind\\PDBBind_processed\\{name}\\{name}_protein_processed.pdb',
                        f'C:\\data\\LGDrugAI\\ligand_prep\\{name}_ligand_prep.sdf',
                                  pocket_only=True)
    converter = GraphFormatConvertor()
    G = converter.convert_nx_to_pyg(g)
    torch.save(G, f'C:\\database\\PDBBind\\PDBBind_pyg_feature\\{name}_pyg.pt')
    save_pdb_df_to_pdb(df, f'C:\\database\\PDBBind\\PDBBind_pyg_feature\\{name}_pocket.pdb')
    with open('C:\\tmp\\check.txt', 'a') as f:
        check = isclose(G['coords'], G['rdkit_atom_feature'][:, -3:], atol=0.1).all()
        f.write(f'{name} {check}\n')
    # except:
    #     with open('C:\\tmp\\check.txt', 'a') as f:
    #         f.write(f'{name} Fail\n')


if __name__ == '__main__':
    # from multiprocessing import Pool
    # from multiprocessing import freeze_support
    # freeze_support()
    # import os
    # from glob import glob
    # from torch import isclose
    # from wcode.protein.biodf import save_pdb_df_to_pdb
    #
    # names = os.listdir('C:\\database\\PDBBind\\PDBBind_processed')
    # complete_names = glob('C:\\database\\PDBBind\\PDBBind_pyg_feature\\*.pt')
    # complete_names = [os.path.basename(n).split('_')[0] for n in complete_names]
    # names = [n for n in names if n not in complete_names]
    # # names = ['1ai6']
    #
    # # print(len(names))
    # pool = Pool(8)
    # for name in names:
    #     pool.apply_async(func=func, args=(name,))
    # pool.close()
    # pool.join()

    g = construct_graph(f'C:\\database\\PDBBind\\PDBBind_processed\\185l\\185l_protein_processed.pdb',
                         f'C:\\data\\LGDrugAI\\ligand_prep\\\\185l_ligand_prep.sdf')
    converter = GraphFormatConvertor()
    G = converter.convert_to_hetero_graph(g[0])

    # torch.save(G, f'C:\\database\\PDBBind\\PDBBind_processed\\185l\\185l_pyg.pt')

    from torch_geometric.nn import GATConv, Linear, to_hetero


    class GAT(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
            self.lin1 = Linear(-1, hidden_channels)
            self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
            self.lin2 = Linear(-1, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index) + self.lin1(x)
            x = x.relu()
            x = self.conv2(x, edge_index) + self.lin2(x)
            return x


    model = GAT(hidden_channels=64, out_channels=5)
    model = to_hetero(model, G.metadata(), aggr='sum')
    out = model(G.x_dict, G.edge_index_dict)
    print(out)