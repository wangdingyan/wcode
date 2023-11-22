import pandas as pd
from scipy.spatial.distance import pdist, squareform
from wcode.protein.convert import filter_dataframe
import networkx as nx
import numpy as np

# https://github.com/a-r-j/graphein/blob/master/graphein/protein/graphs.py
########################################################################################################################

########################################################################################################################


def compute_distmat(pdb_df: pd.DataFrame) -> pd.DataFrame:
    '''
    pdb_df -> [N_Atom, N_Atom]
    '''
    if (
        not pd.Series(["x_coord", "y_coord", "z_coord"])
        .isin(pdb_df.columns)
        .all()
    ):
        raise ValueError(
            "Dataframe must contain columns ['x_coord', 'y_coord', 'z_coord']"
        )
    eucl_dists = pdist(
        pdb_df[["x_coord", "y_coord", "z_coord"]], metric="euclidean"
    )
    eucl_dists = pd.DataFrame(squareform(eucl_dists))
    eucl_dists.index = pdb_df.index
    eucl_dists.columns = pdb_df.index

    return eucl_dists


def get_interacting_atoms(
    angstroms: float, distmat: pd.DataFrame
) -> np.ndarray:

    return np.where(distmat <= angstroms)


class EDGE_CONSTRUCTION_FUNCS():
    def __init__(self, **kwargs):
        self.long_interaction_threshold = 10 if "long_interaction_threshold" not in kwargs else kwargs["long_interaction_threshold"]
        self.threshold = 5 if "threshold" not in kwargs else kwargs["threshold"]

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


