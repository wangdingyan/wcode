from tempfile import TemporaryDirectory
import os
from rdkit import Chem
from wcode.protein.convert import filter_dataframe, save_pdb_df_to_pdb
from wcode.protein.constant import *
from wcode.protein.graph.graph_distance import *
from wcode.mol._atom import featurize_atom_one_hot, featurize_atom
import networkx as nx
import numpy as np
import pandas as pd
from copy import deepcopy
from rdkit.Chem import AllChem
from wcode.mol._bond import get_bond_features

# https://github.com/a-r-j/graphein/blob/master/graphein/protein/graphs.py
########################################################################################################################

########################################################################################################################


class EDGE_CONSTRUCTION_FUNCS():
    def __init__(self, **kwargs):
        self.long_interaction_threshold = 2 if "long_interaction_threshold" not in kwargs else kwargs["long_interaction_threshold"]
        self.threshold = 5 if "threshold" not in kwargs else kwargs["threshold"]

        self.tolerance = 0.56 if "tolerance" not in kwargs else kwargs["tolerance"]

        self.ligand_smiles = None if "ligand_smiles" not in kwargs else kwargs["ligand_smiles"]

    def add_edges_with_distance_threshold(self, G: nx.Graph):

        pdb_df = filter_dataframe(
            G.graph["pdb_df"], "node_id", list(G.nodes()), True
        )
        dist_mat = compute_distmat(pdb_df)
        interacting_nodes = get_interacting_atoms(self.threshold, distmat=dist_mat)
        interacting_nodes = list(zip(interacting_nodes[0], interacting_nodes[1]))

        count = 0
        for a1, a2 in interacting_nodes:
            n1 = G.graph["pdb_df"].loc[a1, "node_id"]
            n2 = G.graph["pdb_df"].loc[a2, "node_id"]
            n1_chain = G.graph["pdb_df"].loc[a1, "chain_id"]
            n2_chain = G.graph["pdb_df"].loc[a2, "chain_id"]
            n1_position = G.graph["pdb_df"].loc[a1, "residue_number"]
            n2_position = G.graph["pdb_df"].loc[a2, "residue_number"]

            # condition_1 = n1_chain == n2_chain
            condition_2 = (
                abs(n1_position - n2_position) < self.long_interaction_threshold
            )

            #if not (condition_1 and condition_2):
            if not condition_2:
                count += 1
                if G.has_edge(n1, n2):
                    G.edges[n1, n2]["kind"].add("distance_threshold")
                else:
                    G.add_edge(n1, n2, kind={"distance_threshold"})

    def add_covalent_edges(self, G: nx.Graph) -> nx.Graph:
        pdb_df = filter_dataframe(
            G.graph["pdb_df"], "node_id", list(G.nodes()), True
        )
        with TemporaryDirectory() as t:
            protein_file_name = os.path.join(t, 'protein_tmp.pdb')
            df_file_name = os.path.join(t, 'dataframe.xlsx')
            save_pdb_df_to_pdb(pdb_df, protein_file_name)
            pdb_df.to_excel(df_file_name)
            mol = Chem.MolFromPDBFile(protein_file_name, sanitize=False)
            mol_idx_to_graph_nodeid = {atom.GetIdx(): nodeid for atom, nodeid in zip(mol.GetAtoms(), pdb_df['node_id'])}

        for bond in mol.GetBonds():
            n1, n2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            n1 = mol_idx_to_graph_nodeid[n1]
            n2 = mol_idx_to_graph_nodeid[n2]
            if G.has_edge(n1, n2):
                if not "covalent" in G.edges[n1, n2]["kind"]:
                    G.edges[n1, n2]["kind"].add("covalent")
                G.edges[n1, n2]["bond_feature"] = get_bond_features(bond)
            else:
                G.add_edge(n1, n2, kind={"covalent"}, bond_feature=get_bond_features(bond))
        return G

    def add_hetatm_covalent_edges(self, G: nx.Graph) -> nx.Graph:
        if len(G.graph['keep_hets']) == 0:
            return G
        else:
            for h_group in G.graph['keep_hets']:
                df_het = deepcopy(G.graph['pdb_df'])
                df_het = df_het.loc[df_het['record_name'] == 'HETATM']
                with TemporaryDirectory() as t:
                    het_file_name = os.path.join(t, h_group+'.pdb')
                    save_pdb_df_to_pdb(df_het, het_file_name)
                    mol = Chem.MolFromPDBFile(het_file_name, sanitize=False)
                    template = Chem.MolFromSmiles(self.ligand_smiles.replace('/', '').replace('\\', ''))
                    mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
                    mol_idx_to_graph_nodeid = {atom.GetIdx(): nodeid for atom, nodeid in zip(mol.GetAtoms(), df_het['node_id'])}

                for atom in mol.GetAtoms():
                    n = atom.GetIdx()
                    n = mol_idx_to_graph_nodeid[n]
                    feature = featurize_atom(atom)
                    feature_one_hot = featurize_atom_one_hot(atom)
                    G.nodes[n]['rdkit_atom_feature'] = feature
                    G.nodes[n]['rdkit_atom_feature_onehot'] = feature_one_hot

                for bond in mol.GetBonds():
                    n1, n2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    n1 = mol_idx_to_graph_nodeid[n1]
                    n2 = mol_idx_to_graph_nodeid[n2]
                    if G.has_edge(n1, n2):
                        if not "covalent" in G.edges[n1, n2]["kind"]:
                            G.edges[n1, n2]["kind"].add("covalent")
                        G.edges[n1, n2]["bond_feature"] = get_bond_features(bond)
                    else:
                        G.add_edge(n1, n2, kind={"covalent"}, bond_feature=get_bond_features(bond))
            return G

    # def add_atm_covalent_edges(self, G: nx.Graph) -> nx.Graph:
    #     dist_mat = compute_distmat(G.graph["pdb_df"])
    #
    #     G.graph["pdb_df"] = assign_bond_states_to_dataframe(G.graph["pdb_df"])
    #     G.graph["pdb_df"] = assign_covalent_radii_to_dataframe(G.graph["pdb_df"])
    #
    #     covalent_radius_distance_matrix = np.add(
    #         np.array(G.graph["pdb_df"]["covalent_radius"]).reshape(-1, 1),
    #         np.array(G.graph["pdb_df"]["covalent_radius"]).reshape(1, -1),
    #     )
    #
    #     covalent_radius_distance_matrix = (
    #             covalent_radius_distance_matrix + self.tolerance
    #     )
    #
    #     dist_mat = dist_mat[dist_mat > 0.4]
    #     t_distmat = dist_mat[dist_mat < covalent_radius_distance_matrix]
    #     G.graph["atomic_adj_mat"] = np.nan_to_num(t_distmat)
    #     inds = zip(*np.where(~np.isnan(t_distmat)))
    #     for i in inds:
    #         length = t_distmat[i[0]][i[1]]
    #         node_1 = G.graph["pdb_df"]["node_id"][i[0]]
    #         node_2 = G.graph["pdb_df"]["node_id"][i[1]]
    #         chain_1 = G.graph["pdb_df"]["chain_id"][i[0]]
    #         chain_2 = G.graph["pdb_df"]["chain_id"][i[1]]
    #         # Check nodes are in graph
    #         if not (G.has_node(node_1) and G.has_node(node_2)):
    #             continue
    #         # Check atoms are in the same chain
    #         if chain_1 != chain_2:
    #             continue
    #         if G.has_edge(node_1, node_2):
    #             G.edges[node_1, node_2]["kind"].add("covalent")
    #             G.edges[node_1, node_2]["bond_length"] = length
    #         else:
    #             G.add_edge(node_1, node_2, kind={"covalent"}, bond_length=length)
    #
    #     return G


# def assign_bond_states_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
#     naive_bond_states = pd.Series(df["atom_name"].map(DEFAULT_BOND_STATE))
#     ss = (
#         pd.DataFrame(RESIDUE_ATOM_BOND_STATE)
#         .unstack()
#         .rename_axis(("residue_name", "atom_name"))
#         .rename("atom_bond_state")
#     )
#     df = df.join(ss, on=["residue_name", "atom_name"])
#     df = df.fillna(value={"atom_bond_state": naive_bond_states})
#     return df


# def assign_covalent_radii_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
#     df["covalent_radius"] = df["atom_bond_state"].map(COVALENT_RADII)
#     return df
#
#
# def add_atm_bond_order(G: nx.Graph) -> nx.Graph:
#     for u, v, a in G.edges(data=True):
#         if G.nodes[u]['record_name'] != 'ATOM' or G.nodes[v]['record_name'] != 'ATOM':
#             continue
#         if 'covalent' not in G.edges[u, v]["kind"]:
#             continue
#         atom_a = G.nodes[u]["element_symbol"]
#         atom_b = G.nodes[v]["element_symbol"]
#
#         # Assign bonds with hydrogens to 1
#         if atom_a == "H" or atom_b == "H":
#             G.edges[u, v]["kind"].add("SINGLE")
#         # If not, we need to identify the bond type from the bond length
#         else:
#             query = f"{atom_a}-{atom_b}"
#             # We need this try block as the dictionary keys may be X-Y, whereas
#             # the query we construct may be Y-X
#             try:
#                 identify_bond_type_from_mapping(G, u, v, a, query)
#             except KeyError:
#                 query = f"{atom_b}-{atom_a}"
#                 try:
#                     identify_bond_type_from_mapping(G, u, v, a, query)
#                 except KeyError:
#                     print(
#                         f"Could not identify bond type for {query}. Adding a \
#                             single bond."
#                     )
#                     G.edges[u, v]["kind"].add("SINGLE")
#
#     return G
#
#
# def identify_bond_type_from_mapping(
#     G: nx.Graph,
#     u: str,
#     v: str,
#     a: Dict,
#     query: str
# ):
#     allowable_order = BOND_ORDERS[query]
#     # If max double, compare the length to the double watershed distance, w_sd,
#     # else assign single
#     if len(allowable_order) == 2:
#         if a["bond_length"] < BOND_LENGTHS[query]["w_sd"]:
#             G.edges[u, v]["kind"].add("DOUBLE")
#         else:
#             G.edges[u, v]["kind"].add("SINGLE")
#     else:
#         # If max triple, compare the length to the triple watershed distance,
#         # w_dt, then double, else assign single
#         if a["bond_length"] < BOND_LENGTHS[query]["w_dt"]:
#             G.edges[u, v]["kind"].add("TRIPLE")
#         elif a["bond_length"] < BOND_LENGTHS[query]["w_sd"]:
#             G.edges[u, v]["kind"].add("DOUBLE")
#         else:
#             G.edges[u, v]["kind"].add("SINGLE")
#     return G
#
#
def add_distance_to_edges(G: nx.Graph) -> nx.Graph:
    '''
    + G.graph["dist_mat"]
    + d["distance"]
    '''

    dist_mat = compute_distmat(G.graph["pdb_df"])
    G.graph["dist_mat"] = dist_mat

    mat = np.where(nx.to_numpy_array(G), dist_mat, 0)
    node_map = {n: i for i, n in enumerate(G.nodes)}
    for u, v, d in G.edges(data=True):
        d["distance"] = [mat[node_map[u], node_map[v]]]
    return G
