import pandas as pd
import networkx as nx
import numpy as np
import os
from copy import deepcopy
from tempfile import TemporaryDirectory
from wcode.protein.constant import STANDARD_RESI_NAMES, PROTEIN_ATOMS, ATOM_SYMBOL
from wcode.protein.convert import save_pdb_df_to_pdb
from wcode.mol._atom import featurize_atom, featurize_atom_one_hot
from rdkit import Chem

########################################################################################################################

########################################################################################################################
def add_nodes_to_graph(
    G: nx.Graph,
    protein_df = None,
    verbose: bool = False,
) -> nx.Graph:

    if protein_df is None:
        protein_df: pd.DataFrame = G.graph["pdb_df"]

    with TemporaryDirectory() as temp_dir:
        protein_file_name = os.path.join(temp_dir, "temp_protein.pdb")
        save_pdb_df_to_pdb(protein_df, protein_file_name)
        mol = Chem.MolFromPDBFile(protein_file_name)
        rdkit_atom_feature = []
        rdkit_atom_feature_one_hot = []
        for atom in mol.GetAtoms():
            rdkit_atom_feature.append(str(featurize_atom(atom).tolist()))
            rdkit_atom_feature_one_hot.append(str(featurize_atom_one_hot(atom).tolist()))

    # Assign intrinsic node attributes
    chain_id = protein_df["chain_id"].apply(str)

    # residue type
    residue_name = protein_df["residue_name"]
    residue_encoding = one_of_k_encoding_unk(STANDARD_RESI_NAMES, False)
    residue_name_one_hot = residue_name.apply(residue_encoding)

    # residue number
    residue_number = protein_df["residue_number"]  # .apply(str)

    # coordination
    coords = np.asarray(protein_df[["x_coord", "y_coord", "z_coord"]])

    # b_factor
    b_factor = protein_df["b_factor"]

    # protein atom type
    atom_type = protein_df["atom_name"]
    atom_encoding = one_of_k_encoding_unk(PROTEIN_ATOMS, True)
    atom_type_one_hot = atom_type.apply(atom_encoding)

    # node_id
    nodes = protein_df["node_id"]

    # element symbol
    element_symbol = protein_df["element_symbol"]
    element_encoding = one_of_k_encoding_unk(ATOM_SYMBOL, False)
    element_symbol_one_hot = element_symbol.apply(element_encoding)

    RECORD_NAME = ['ATOM', 'HETATM']
    record_name = protein_df["record_name"]
    record_encoding = one_of_k_encoding_unk(RECORD_NAME, True)
    record_encoding_one_hot = record_name.apply(record_encoding)

    G.add_nodes_from(nodes)

    # Set intrinsic node attributes
    nx.set_node_attributes(G, dict(zip(nodes, chain_id)), "chain_id")

    nx.set_node_attributes(G, dict(zip(nodes, residue_name)), "residue_name")
    nx.set_node_attributes(G, dict(zip(nodes, residue_name_one_hot)), "residue_name_one_hot")

    nx.set_node_attributes(
        G, dict(zip(nodes, residue_number)), "residue_number"
    )
    nx.set_node_attributes(G, dict(zip(nodes, atom_type)), "atom_type")
    nx.set_node_attributes(G, dict(zip(nodes, atom_type_one_hot)), "atom_type_one_hot")

    nx.set_node_attributes(G, dict(zip(nodes, element_symbol)), "element_symbol")
    nx.set_node_attributes(G, dict(zip(nodes, element_symbol_one_hot)), "element_symbol_one_hot")

    nx.set_node_attributes(G, dict(zip(nodes, coords)), "coords")
    nx.set_node_attributes(G, dict(zip(nodes, b_factor)), "b_factor")

    nx.set_node_attributes(G, dict(zip(nodes, record_name)), "record_name")
    nx.set_node_attributes(G, dict(zip(nodes, record_encoding_one_hot)), "record_symbol_one_hot")

    nx.set_node_attributes(G, dict(zip(nodes, rdkit_atom_feature)), "rdkit_atom_feature")
    nx.set_node_attributes(G, dict(zip(nodes, rdkit_atom_feature_one_hot)), "rdkit_atom_feature_onehot")

    if verbose:
        print(G)
        print(G.nodes())

    return G

class one_of_k_encoding_unk():
    def __init__(self,
                 allowable_set,
                 append_UNK=True):
        self.allowable_set = deepcopy(allowable_set)
        self.allowable_set = allowable_set
        if append_UNK:
            self.allowable_set.append('UNK')

    def __call__(self, x):
        if x not in self.allowable_set:
            x = self.allowable_set[-1]
        return np.array([x == s for s in self.allowable_set]).astype(np.float32)