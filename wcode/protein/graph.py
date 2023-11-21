import os
import pandas as pd
from pathlib import Path
from .convert import read_pdb_to_dataframe, filter_dataframe #, three_to_one_with_mods
from .constant import BACKBONE_ATOMS
from typing import Optional
import networkx as nx
import numpy as np

# https://github.com/a-r-j/graphein/blob/master/graphein/protein/graphs.py
########################################################################################################################


def construct_graph(path):
    if path is not None and isinstance(path, Path):
        path = os.fsdecode(path)
    df = read_pdb_to_dataframe(path)

########################################################################################################################


# def initialise_graph_with_metadata(
#     protein_df: pd.DataFrame,
#     granularity = 'atom',
#     path: Optional[str] = None,
# ) -> nx.Graph:
#
#     if path is not None and isinstance(path, Path):
#         path = os.fsdecode(path)
#
#     G = nx.Graph(
#         path=path,
#         chain_ids=list(protein_df["chain_id"].unique()),
#         pdb_df=protein_df,
#         rgroup_df=compute_rgroup_dataframe(protein_df),
#         coords=np.asarray(protein_df[["x_coord", "y_coord", "z_coord"]]),
#     )
#
#
#     # Add Sequences to graph metadata
#     for c in G.graph["chain_ids"]:
#         if granularity == "atom":
#             sequence = (
#                 protein_df.loc[
#                     (protein_df["chain_id"] == c)
#                     & (protein_df["atom_name"] == "CA")
#                 ]["residue_name"]
#                 .apply(three_to_one_with_mods)
#                 .str.cat()
#             )
#         else:
#             sequence = (
#                 protein_df.loc[protein_df["chain_id"] == c]["residue_name"]
#                 .apply(three_to_one_with_mods)
#                 .str.cat()
#             )
#         G.graph[f"sequence_{c}"] = sequence
#     return G
#
#
# def compute_rgroup_dataframe(pdb_df: pd.DataFrame) -> pd.DataFrame:
#     return filter_dataframe(pdb_df, "atom_name", BACKBONE_ATOMS, False)