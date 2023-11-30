import copy
import os
import pandas as pd
from pathlib import Path
from wcode.protein.convert import read_pdb_to_dataframe, filter_dataframe, save_pdb_df_to_pdb
from wcode.protein.constant import BACKBONE_ATOMS, RESI_THREE_TO_1
from wcode.protein.graph.graph_nodes import add_nodes_to_graph
from wcode.protein.graph.graph_edge import add_distance_to_edges, EDGE_CONSTRUCTION_FUNCS
from rdkit import Chem

import networkx as nx
import numpy as np

# https://github.com/a-r-j/graphein/blob/master/graphein/protein/graphs.py
########################################################################################################################


def construct_graph(protein_path,
                    ligand_path=None,
                    compute_edge_funcs=None,
                    keep_hets=[],
                    smiles=None,
                    pocket_only=False,
                    verbose=False):

    keep_hets = copy.deepcopy(keep_hets)
    if ligand_path:
        keep_hets.append('LIG')
        output_path, ligand_smiles = merge_protein_ligand_file(protein_path, ligand_path)
    if smiles:
        ligand_smiles = smiles

    if compute_edge_funcs is None:
        compute_edge_funcs = ["EDGE_CONSTRUCTION_FUNCS(threshold=4.0).add_edges_with_distance_threshold",
                              "EDGE_CONSTRUCTION_FUNCS().add_covalent_edges"]
    if ligand_smiles != None:
        compute_edge_funcs.append(f"EDGE_CONSTRUCTION_FUNCS(ligand_smiles='{ligand_smiles}').add_hetatm_covalent_edges")
    if ligand_path is None:
        output_path = protein_path

    df = read_pdb_to_dataframe(output_path,
                               keep_hets=keep_hets,
                               pocket_only=pocket_only)

    g = initialise_graph_with_metadata(protein_df=df,
                                       keep_hets=keep_hets)

    g = add_nodes_to_graph(g, verbose=verbose)

    for f in compute_edge_funcs:
        eval(f)(g)
    g = add_distance_to_edges(g)
    return g

########################################################################################################################

def initialise_graph_with_metadata(
    protein_df: pd.DataFrame,
    keep_hets,
    granularity='atom',
) -> nx.Graph:

    G = nx.Graph(
        chain_ids=list(protein_df["chain_id"].unique()),
        pdb_df=protein_df,
        rgroup_df=compute_rgroup_dataframe(protein_df),
        coords=np.asarray(protein_df[["x_coord", "y_coord", "z_coord"]]),
        keep_hets=keep_hets
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


def merge_protein_ligand_file(protein_file,
                              ligand_file,
                              output_path=None):
    if output_path is None:
        output_path = protein_file.replace('.pdb', '_merge.pdb')
    ligand_mol = next(Chem.SDMolSupplier(ligand_file))
    ligand_smiles = Chem.MolToSmiles(ligand_mol)
    protein_mol = Chem.MolFromPDBFile(protein_file)
    merge_mol = Chem.CombineMols(protein_mol, ligand_mol)
    Chem.MolToPDBFile(merge_mol, output_path)
    with open(output_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith('ATOM'):
                res_num = int(line[22:26])

        res_num += 1
        res_num = list(str(res_num).rjust(4))
        for i, line in enumerate(lines):
            if line.startswith('HETATM'):
                line = list(line)
                line[21] = 'Z'
                line[17], line[18], line[19] = 'L', 'I', 'G'
                line[22], line[23], line[24], line[25] = res_num
                line = "".join(line)
                lines[i] = line
    with open(output_path, 'w') as f:
        for line in lines:
            f.write(line)
    return output_path, ligand_smiles


if __name__ == '__main__':
    # g = construct_graph('/mnt/d/tmp/5p21.pdb',
    #                     verbose=True,
    #                     smiles='O=P(O)(O)NP(=O)(O)OP(=O)(O)OCC1C(O)C(O)C(O1)n(cn2)c(c23)nc(N)[nH]c3=O',
    #                     keep_hets=['GNP'],
    #                     pocket_only=True)
    # print(g)
    # print(list(g.edges))
    # print(g)
    # for n, data in g.nodes(data=True):
    #     print(f"节点 {n} 的属性为: {data}")
    # for u, v, data in g.edges(data=True):
    #     print(f"边 ({u}, {v}) 的属性为: {data}")
    import torch
    # data = {"node_id": list(G.nodes())}
    # G = nx.convert_node_labels_to_integers(G)
    #
    # print(list(G.edges(data=True)))
    # for u, v, data in G.edges(data=True):
    #     print(f"边 ({u}, {v}) 的属性为: {data}")

    g = construct_graph('D:\\PDBBind\\PDBBind_processed\\1a0q\\1a0q_protein_processed.pdb',
                    'D:\\PDBBind\\PDBBind_processed\\1a0q\\1a0q_ligand.sdf',
                        pocket_only=True)
    # for u, v, data in g.edges(data=True):
    #     print(f"边 ({u}, {v}) 的属性为: {data}")
    for n, data in g.nodes(data=True):
        print(f"节点 {n} 的属性为: {data}")

