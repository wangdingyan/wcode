import pandas as pd
import networkx as nx
import tempfile
from typing import Any, Dict, Optional
import numpy as np
import os
from copy import deepcopy
from tempfile import TemporaryDirectory
from wcode.protein.constant import STANDARD_RESI_NAMES, PROTEIN_ATOMS, ATOM_SYMBOL, DSSP_COLS, STANDARD_AMINO_ACID_MAPPING_1_TO_3, DSSP_SS
from wcode.protein.biodf import save_pdb_df_to_pdb
from wcode.mol._atom import featurize_atom, featurize_atom_one_hot
from wcode.utils.dependency import is_tool
from wcode.protein.seq.embedding import esm_residue_embedding
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, residue_max_acc
from rdkit import Chem

########################################################################################################################

########################################################################################################################
def add_nodes_to_graph(
    G: nx.Graph,
    dssp=False,
    esm=False,
    protein_df = None,
    verbose: bool = False,
) -> nx.Graph:

    if protein_df is None:
        protein_df: pd.DataFrame = G.graph["pdb_df"]

    with TemporaryDirectory() as temp_dir:
        protein_file_name = os.path.join(temp_dir, "temp_protein.pdb")
        save_pdb_df_to_pdb(protein_df, protein_file_name)
        mol = Chem.MolFromPDBFile(protein_file_name, sanitize=False)
        rdkit_atom_feature = []
        rdkit_atom_feature_one_hot = []
        for atom in mol.GetAtoms():
            rdkit_atom_feature.append(featurize_atom(atom))
            rdkit_atom_feature_one_hot.append(featurize_atom_one_hot(atom))

    # Assign intrinsic node attributes
    chain_id = protein_df["chain_id"].apply(str)
    residue_id = protein_df["residue_id"].apply(str)

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
    nx.set_node_attributes(G, dict(zip(nodes, residue_id)), "residue_id")

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

    if dssp:
        G = rsa(G)
        G = asa(G)
        G = phi(G)
        G = psi(G)
        G = secondary_structure(G)

    if esm:
        G = esm_residue_embedding(G)

    if verbose:
        print(G)
        print(G.nodes())

    return G

class one_of_k_encoding_unk():
    def __init__(self,
                 allowable_set,
                 append_UNK=True):
        self.allowable_set = deepcopy(allowable_set)
        if append_UNK:
            self.allowable_set.append('UNK')

    def __call__(self, x):
        if x not in self.allowable_set:
            x = self.allowable_set[-1]
        return np.array([x == s for s in self.allowable_set]).astype(np.float32)



def parse_dssp_df(dssp: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse ``DSSP`` output to DataFrame.

    :param dssp: Dictionary containing ``DSSP`` output
    :type dssp: Dict[str, Any]
    :return: pd.DataFrame containing parsed ``DSSP`` output
    :rtype: pd.DataFrame
    """
    appender = []
    for k in dssp[1]:
        to_append = []
        y = dssp[0][k]
        chain = k[0]
        residue = k[1]
        # het = residue[0]
        resnum = residue[1]
        icode = residue[2]
        to_append.extend([chain, resnum, icode])
        to_append.extend(y)
        appender.append(to_append)

    return pd.DataFrame.from_records(appender, columns=DSSP_COLS)

def add_dssp_df(
    G: nx.Graph,
) -> nx.Graph:


    executable = "mkdssp"
    assert is_tool(
        executable
    ), "DSSP must be on PATH and marked as an executable"

    with tempfile.TemporaryDirectory() as tmpdirname:


        save_pdb_df_to_pdb(
            G.graph["raw_pdb_df"], os.path.join(tmpdirname, 'temp_dssp.pdb')
        )
        dssp_dict = dssp_dict_from_pdb_file(
            os.path.join(tmpdirname, 'temp_dssp.pdb'), DSSP=executable
        )
        dssp_dict = parse_dssp_df(dssp_dict)
        # Convert 1 letter aa code to 3 letter
        dssp_dict["aa"] = dssp_dict["aa"].map(STANDARD_AMINO_ACID_MAPPING_1_TO_3)
        # Resolve UNKs
        dssp_dict.loc[dssp_dict["aa"] == "UNK", "aa"] = (
            G.graph["pdb_df"]
            .loc[
                G.graph["pdb_df"].residue_number.isin(
                    dssp_dict.loc[dssp_dict["aa"] == "UNK"]["resnum"]
                )
            ]["residue_name"]
            .values
        )

        # Construct node IDs
        dssp_dict["node_id"] = (
                dssp_dict["chain"]
                + ":"
                + dssp_dict["aa"]
                + ":"
                + dssp_dict["resnum"].astype(str)
        )

        dssp_dict.set_index("node_id", inplace=True)
        G.graph["dssp_df"] = dssp_dict

        return G

def add_dssp_feature(G: nx.Graph,
                     feature: str) -> nx.Graph:
    """
    Adds specified amino acid feature as calculated
    by DSSP to every node in a protein graph

    :param G: Protein structure graph to add dssp feature to
    :type G: nx.Graph
    :param feature: string specifying name of DSSP feature to add:
        ``"chain"``,
        ``"resnum"``,
        ``"icode"``,
        ``"aa"``,
        ``"ss"``,
        ``"asa"``,
        ``"phi"``,
        ``"psi"``,
        ``"dssp_index"``,
        ``"NH_O_1_relidx"``,
        ``"NH_O_1_energy"``,
        ``"O_NH_1_relidx"``,
        ``"O_NH_1_energy"``,
        ``"NH_O_2_relidx"``,
        ``"NH_O_2_energy"``,
        ``"O_NH_2_relidx"``,
        ``"O_NH_2_energy"``,
        These names are accessible in the DSSP_COLS list
    :type feature: str
    :return: Protein structure graph with DSSP feature added to nodes
    :rtype: nx.Graph
    """
    if "dssp_df" not in G.graph:
        G = add_dssp_df(G)
    dssp_df = G.graph["dssp_df"]

    assign_feat(G, dict(dssp_df[feature]), feature)
    if feature == 'ss':
        ss_symbol = dssp_df["ss"]
        ss_encoding = one_of_k_encoding_unk(DSSP_SS, True)
        ss_encoding_one_hot = ss_symbol.apply(ss_encoding)
        assign_feat(G, dict(ss_encoding_one_hot), 'ss_onehot')
    return G

def assign_feat(G, feature_dict, feature_name):
    for n, d in G.nodes(data=True):
        residue_id = G.nodes[n]['residue_id']
        if residue_id in feature_dict:
            G.nodes[n][feature_name] = feature_dict[residue_id]



def rsa(G: nx.Graph) -> nx.Graph:
    """
    Adds RSA (relative solvent accessibility) of each residue in protein graph
    as calculated by DSSP.

    :param G: Input protein graph
    :type G: nx.Graph
    :return: Protein graph with rsa values added
    :rtype: nx.Graph
    """

    # Calculate RSA
    try:
        dssp_df = G.graph["dssp_df"]
    except KeyError:
        G = add_dssp_df(G)
        dssp_df = G.graph["dssp_df"]
    dssp_df["max_acc"] = dssp_df["aa"].map(residue_max_acc["Sander"].get)
    dssp_df[["asa", "max_acc"]] = dssp_df[["asa", "max_acc"]].astype(float)
    dssp_df["rsa"] = dssp_df["asa"] / dssp_df["max_acc"]

    G.graph["dssp_df"] = dssp_df

    return add_dssp_feature(G, "rsa")


def asa(G: nx.Graph) -> nx.Graph:
    """
    Adds ASA of each residue in protein graph as calculated by DSSP.

    :param G: Input protein graph
    :type G: nx.Graph
    :return: Protein graph with asa values added
    :rtype: nx.Graph
    """
    return add_dssp_feature(G, "asa")


def phi(G: nx.Graph) -> nx.Graph:
    """
    Adds phi-angles of each residue in protein graph as calculated by DSSP.

    :param G: Input protein graph
    :type G: nx.Graph
    :return: Protein graph with phi-angles values added
    :rtype: nx.Graph
    """
    return add_dssp_feature(G, "phi")


def psi(G: nx.Graph) -> nx.Graph:
    """
    Adds psi-angles of each residue in protein graph as calculated by DSSP.

    :param G: Input protein graph
    :type G: nx.Graph
    :return: Protein graph with psi-angles values added
    :rtype: nx.Graph
    """
    return add_dssp_feature(G, "psi")


def secondary_structure(G: nx.Graph) -> nx.Graph:
    """
    Adds secondary structure of each residue in protein graph
    as calculated by DSSP in the form of a string

    :param G: Input protein graph
    :type G: nx.Graph
    :return: Protein graph with secondary structure added
    :rtype: nx.Graph
    """
    return add_dssp_feature(G, "ss")
