import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils.undirected import to_undirected
from wcode.protein.graph.graph import construct_graph


########################################################################################################################

class GraphFormatConvertor:
    def __init__(self):
        columns = [
            # node feature
            "coords",
            "node_id",
            "residue_name_one_hot",
            "atom_type_one_hot",
            "record_symbol_one_hot",
            "rdkit_atom_feature",
            "rdkit_atom_feature_onehot",
            "node_id"

            # edge feature
            "edge_index",
            "distance",
            "bond_feature",

            # edge index
            "edge_index_covalent",
            "edge_index_distance_threshold",
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
                if key == 'rdkit_atom_feature':
                    with open('C:\\tmp\\check_dimension.txt', 'a') as f:
                        f.write(str(value)+'\n')
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


########################################################################################################################

g = construct_graph('D:\\PDBBind\\PDBBind_processed\\1a0t\\1a0t_protein_processed.pdb',
                    'D:\\PDBBind\\PDBBind_processed\\1a0t\\1a0t_ligand.sdf',
                              pocket_only=True)
converter = GraphFormatConvertor()
G = converter.convert_nx_to_pyg(g)

print(G)
from torch import isclose
print(isclose(G['coords'], G['rdkit_atom_feature'][:, -3:], atol=0.1).all())
