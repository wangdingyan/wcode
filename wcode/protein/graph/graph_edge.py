import pandas as pd
from wcode.protein.convert import filter_dataframe
from wcode.protein.constant import *
from wcode.protein.graph.graph_distance import *
import networkx as nx
import numpy as np

# https://github.com/a-r-j/graphein/blob/master/graphein/protein/graphs.py
########################################################################################################################

########################################################################################################################

class EDGE_CONSTRUCTION_FUNCS():
    def __init__(self, **kwargs):
        self.long_interaction_threshold = 10 if "long_interaction_threshold" not in kwargs else kwargs["long_interaction_threshold"]
        self.threshold = 5 if "threshold" not in kwargs else kwargs["threshold"]

        self.tolerance = 0.56 if "tolerance" not in kwargs else kwargs["tolerance"]

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

            condition_1 = n1_chain == n2_chain
            condition_2 = (
                abs(n1_position - n2_position) < self.long_interaction_threshold
            )

            if not (condition_1 and condition_2):
                count += 1
                if G.has_edge(n1, n2):
                    G.edges[n1, n2]["kind"].add("distance_threshold")
                else:
                    G.add_edge(n1, n2, kind={"distance_threshold"})

    def add_atomic_edges(self,  G: nx.Graph) -> nx.Graph:
        dist_mat = compute_distmat(G.graph["pdb_df"])

        G.graph["pdb_df"] = assign_bond_states_to_dataframe(G.graph["pdb_df"])
        G.graph["pdb_df"] = assign_covalent_radii_to_dataframe(G.graph["pdb_df"])

        covalent_radius_distance_matrix = np.add(
            np.array(G.graph["pdb_df"]["covalent_radius"]).reshape(-1, 1),
            np.array(G.graph["pdb_df"]["covalent_radius"]).reshape(1, -1),
        )

        covalent_radius_distance_matrix = (
                covalent_radius_distance_matrix + self.tolerance
        )

        dist_mat = dist_mat[dist_mat > 0.4]
        t_distmat = dist_mat[dist_mat < covalent_radius_distance_matrix]
        G.graph["atomic_adj_mat"] = np.nan_to_num(t_distmat)
        inds = zip(*np.where(~np.isnan(t_distmat)))
        for i in inds:
            length = t_distmat[i[0]][i[1]]
            node_1 = G.graph["pdb_df"]["node_id"][i[0]]
            node_2 = G.graph["pdb_df"]["node_id"][i[1]]
            chain_1 = G.graph["pdb_df"]["chain_id"][i[0]]
            chain_2 = G.graph["pdb_df"]["chain_id"][i[1]]
            # Check nodes are in graph
            if not (G.has_node(node_1) and G.has_node(node_2)):
                continue
            # Check atoms are in the same chain
            if chain_1 != chain_2:
                continue
            if G.has_edge(node_1, node_2):
                G.edges[node_1, node_2]["kind"].add("covalent")
                G.edges[node_1, node_2]["bond_length"] = length
            else:
                G.add_edge(node_1, node_2, kind={"covalent"}, bond_length=length)

        return G


def assign_bond_states_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    naive_bond_states = pd.Series(df["atom_name"].map(DEFAULT_BOND_STATE))
    ss = (
        pd.DataFrame(RESIDUE_ATOM_BOND_STATE)
        .unstack()
        .rename_axis(("residue_name", "atom_name"))
        .rename("atom_bond_state")
    )
    df = df.join(ss, on=["residue_name", "atom_name"])
    df = df.fillna(value={"atom_bond_state": naive_bond_states})
    return df


def assign_covalent_radii_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["covalent_radius"] = df["atom_bond_state"].map(COVALENT_RADII)
    return df


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
        d["distance"] = mat[node_map[u], node_map[v]]
    return G


