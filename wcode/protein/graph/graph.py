import os
import pandas as pd
from pathlib import Path
from wcode.protein.convert import read_pdb_to_dataframe, filter_dataframe #, three_to_one_with_mods
from wcode.protein.constant import BACKBONE_ATOMS, RESI_THREE_TO_1
from wcode.protein.graph.graph_nodes import add_nodes_to_graph
from wcode.protein.graph.graph_edge import add_distance_to_edges, EDGE_CONSTRUCTION_FUNCS
import networkx as nx
import numpy as np

# https://github.com/a-r-j/graphein/blob/master/graphein/protein/graphs.py
########################################################################################################################


def construct_graph(path,
                    compute_edge_funcs = ["EDGE_CONSTRUCTION_FUNCS(threshold=3.0).add_edges_with_distance_threshold"],
                    verbose=False):
    if path is not None and isinstance(path, Path):
        path = os.fsdecode(path)
    df = read_pdb_to_dataframe(path)
    g = initialise_graph_with_metadata(protein_df=df)
    g = add_nodes_to_graph(g, verbose=verbose)
    for f in compute_edge_funcs:
        eval(f)(g)
    g = add_distance_to_edges(g)
    return g


########################################################################################################################


def initialise_graph_with_metadata(
    protein_df: pd.DataFrame,
    granularity = 'atom'
) -> nx.Graph:

    G = nx.Graph(
        chain_ids=list(protein_df["chain_id"].unique()),
        pdb_df=protein_df,
        rgroup_df=compute_rgroup_dataframe(protein_df),
        coords=np.asarray(protein_df[["x_coord", "y_coord", "z_coord"]]),
    )

    # Add Sequences to graph metadata
    for c in G.graph["chain_ids"]:
        if granularity == "atom":
            sequence = (
                protein_df.loc[
                    (protein_df["chain_id"] == c)
                    & (protein_df["atom_name"] == "CA")
                ]["residue_name"]
                .apply(three_to_one_with_mods)
                .str.cat()
            )
        else:
            sequence = (
                protein_df.loc[protein_df["chain_id"] == c]["residue_name"]
                .apply(three_to_one_with_mods)
                .str.cat()
            )
        G.graph[f"sequence_{c}"] = sequence
    return G


def compute_rgroup_dataframe(pdb_df: pd.DataFrame) -> pd.DataFrame:
    return filter_dataframe(pdb_df, "atom_name", BACKBONE_ATOMS, False)


def three_to_one_with_mods(res):
    return RESI_THREE_TO_1[res]


if __name__ == '__main__':
    g = construct_graph('/mnt/d/tmp/3rab.pdb', verbose=True)
    # print(g)
    # print(list(g.edges))
    for u, v, data in g.edges(data=True):
        print(f"边 ({u}, {v}) 的属性为: {data}")