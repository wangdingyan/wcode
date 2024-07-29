import copy
import numpy as np
import pandas as pd
import networkx as nx
from biopandas.pdb import PandasPdb
from wcode.protein.biodf import read_pdb_to_dataframe, compute_rgroup_dataframe
from wcode.protein.constant import RESI_THREE_TO_1
from wcode.protein.graph.graph_nodes import add_nodes_to_graph
from wcode.protein.graph.graph_edge import add_distance_to_edges, EDGE_CONSTRUCTION_FUNCS, add_edge_vector
from wcode.pl.merge import merge_protein_ligand_file


# https://github.com/a-r-j/graphein/blob/master/graphein/protein/graphs.py
########################################################################################################################


def construct_graph(protein_path,
                    ligand_path=None,
                    compute_edge_funcs=None,
                    keep_hets=[],
                    smiles=None,
                    granularity='atom',
                    dssp=False,
                    esm=False,
                    pocket_only=False,
                    verbose=False):

    keep_hets = copy.deepcopy(keep_hets)
    ligand_smiles = None
    if ligand_path:
        keep_hets.append('LIG')
        output_path, ligand_smiles = merge_protein_ligand_file(protein_path, ligand_path)

    if smiles:
        ligand_smiles = smiles

    if compute_edge_funcs is None:
        compute_edge_funcs = ["EDGE_CONSTRUCTION_FUNCS(threshold=4.0).add_edges_with_distance_threshold",
                              "EDGE_CONSTRUCTION_FUNCS().add_covalent_edges"]
    if ligand_smiles != None:
        compute_edge_funcs.append(f"EDGE_CONSTRUCTION_FUNCS(ligand_smiles=r'{ligand_smiles}').add_hetatm_covalent_edges")

    if ligand_path is None:
        output_path = protein_path

    raw_df = PandasPdb().read_pdb(output_path).get_model(1)
    raw_df = pd.concat([raw_df.df["ATOM"], raw_df.df["HETATM"]])

    df = read_pdb_to_dataframe(output_path,
                               keep_hets=keep_hets,
                               pocket_only=pocket_only,
                               granularity=granularity)

    g = initialise_graph_with_metadata(protein_df=df,
                                       raw_pdb_df=raw_df,
                                       keep_hets=keep_hets)

    g = add_nodes_to_graph(g,
                           verbose=verbose,
                           dssp=dssp,
                           esm=esm)

    for f in compute_edge_funcs:
        eval(f)(g)
    g = add_distance_to_edges(g)
    g = add_edge_vector(g)

    return g, df


def nxg_to_df(g):
    output_df = {'record_name': [],
                 'atom_number': [],
                 'blank_1':[],
                 'atom_name': [],
                 'alt_loc': [],
                 'residue_name': [],
                 'blank_2': [],
                 'chain_id': [],
                 'residue_number': [],
                 'insertion': [],
                 'blank_3': [],
                 'x_coord': [],
                 'y_coord': [],
                 'z_coord': [],
                 'occupancy': [],
                 'b_factor': [],
                 'blank_4': [],
                 'segment_id': [],
                 'element_symbol': [],
                 'charge': [],
                 'line_idx': [],
                 'model_id': [],
                 'node_id': [],
                 'residue_id': []}

    for i, (n, data) in enumerate(g.nodes(data=True)):
        chain_id, residue_name, residue_number, atom_type = data['node_id'].split(':')
        output_df['record_name'].append(data['record_name'])
        output_df['atom_number'].append(i+1)
        output_df['blank_1'].append('')
        output_df['atom_name'].append(atom_type)
        output_df['alt_loc'].append('')
        output_df['residue_name'].append(residue_name)
        output_df['blank_2'].append('')
        output_df['chain_id'].append(chain_id)
        output_df['residue_number'].append(data['residue_number'])
        output_df['insertion'].append('')
        output_df['blank_3'].append('')
        output_df['x_coord'].append(data['coords'][0])
        output_df['y_coord'].append(data['coords'][1])
        output_df['z_coord'].append(data['coords'][2])
        output_df['occupancy'].append(1.00)
        output_df['b_factor'].append(data['b_factor'])
        output_df['blank_4'].append('')
        output_df['segment_id'].append('0.0')
        output_df['element_symbol'].append(data['element_symbol'])
        output_df['charge'].append(np.nan)
        output_df['line_idx'].append(i)
        output_df['model_id'].append(1)
        output_df['node_id'].append(n)
        output_df['residue_id'].append(':'.join([chain_id, residue_name, residue_number]))

    df = pd.DataFrame(output_df, index=None)
    return df


########################################################################################################################

def initialise_graph_with_metadata(
    protein_df: pd.DataFrame,
    raw_pdb_df: pd.DataFrame,
    keep_hets,
    granularity='atom',
) -> nx.Graph:

    G = nx.Graph(
        chain_ids=list(protein_df["chain_id"].unique()),
        pdb_df=protein_df,
        raw_pdb_df=raw_pdb_df,
        rgroup_df=compute_rgroup_dataframe(protein_df),
        coords=np.asarray(protein_df[["x_coord", "y_coord", "z_coord"]]),
        keep_hets=keep_hets
    )

    # Create graph and assign intrinsic graph-level metadata
    G.graph["node_type"] = granularity

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


def three_to_one_with_mods(res):
    return RESI_THREE_TO_1[res]





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
    # data = {"node_id": list(G.nodes())}
    # G = nx.convert_node_labels_to_integers(G)
    #
    # print(list(G.edges(data=True)))
    # for u, v, data in G.edges(data=True):
    #     print(f"边 ({u}, {v}) 的属性为: {data}")

    g, df = construct_graph('/mnt/d/nutshell/Official/WCODE/sample_data/1a0q_protein_processed_merge.pdb',
                        pocket_only=False,
                        dssp=False,
                        esm=False,
                        granularity='CA',
                        keep_hets=[])
    for n, data in g.nodes(data=True):
        print(f"节点 {n} 的属性为: {data}")
    # for u, v, data in g.edges(data=True):
    #     print(f"边 ({u}, {v}) 的属性为: {data}")
    # print(df['blank_1'][1][0])
    # print(len(df))
    # print(df.columns)
    # print(df['node_id'])
    # print(list(g.nodes(data=True))[65])
    # test_df = nxg_to_df(g)
    # save_pdb_df_to_pdb(test_df, 'C:\\tmp\\20231219.pdb')









