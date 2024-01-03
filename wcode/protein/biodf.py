import os
from pathlib import Path
from typing import Optional, Union, List, Any
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from wcode.protein.graph.graph_distance import compute_distmat, get_interacting_atoms


def read_pdb_to_dataframe(
    path: Optional[Union[str, os.PathLike]] = None,
    model_index: int = 1,
    **kwargs
) -> pd.DataFrame:


    if isinstance(path, Path):
        path = os.fsdecode(path)
    if (
        path.endswith(".pdb")
        or path.endswith(".pdb.gz")
        or path.endswith(".ent")
    ):
        atomic_df = PandasPdb().read_pdb(path)
    else:
        raise ValueError(
            f"File {path} must be either .pdb(.gz), or .ent, not {path.split('.')[-1]}"
        )

    atomic_df = atomic_df.get_model(model_index)
    if len(atomic_df.df["ATOM"]) == 0:
        raise ValueError(f"No model found for index: {model_index}")

    df = pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]])

    df = process_dataframe(df, **kwargs)
    return df


def filter_dataframe(
    dataframe: pd.DataFrame,
    by_column: str,
    list_of_values: List[Any],
    boolean: bool,
) -> pd.DataFrame:

    df = dataframe.copy()
    df = df[df[by_column].isin(list_of_values) == boolean]
    df.reset_index(inplace=True, drop=True)

    return df


def save_pdb_df_to_pdb(
    df: pd.DataFrame,
    path: str,
    gz: bool = False,
    atoms: bool = True,
    hetatms: bool = True,
):

    atom_df = filter_dataframe(df, "record_name", ["ATOM"], boolean=True)
    hetatm_df = filter_dataframe(df, "record_name", ["HETATM"], boolean=True)
    ppd = PandasPdb()
    if atoms:
        ppd.df["ATOM"] = atom_df
    if hetatms:
        ppd.df["HETATM"] = hetatm_df
    ppd.to_pdb(path=path, records=None, gz=gz, append_newline=True)


def process_dataframe(
    protein_df: pd.DataFrame,
    granularity: str = "all",
    chain_selection: str = "all",
    insertions: bool = False,
    alt_locs: bool = False,
    deprotonate: bool = True,
    keep_hets: List[str] = [],
    pocket_only = False,
) -> pd.DataFrame:

    protein_df = label_node_id(protein_df,
                               granularity=granularity,
                               keep_hets=keep_hets)

    atoms = filter_dataframe(
        protein_df,
        by_column="record_name",
        list_of_values=["ATOM"],
        boolean=True,
    )
    hetatms = filter_dataframe(
        protein_df,
        by_column="record_name",
        list_of_values=["HETATM"],
        boolean=True,
    )

    if keep_hets:
        hetatms_to_keep = filter_hetatms(hetatms, keep_hets)
        atoms = pd.concat([atoms] + hetatms_to_keep)

    # Deprotonate structure by removing H atoms
    if deprotonate:
        atoms = deprotonate_structure(atoms)

    # Restrict DF to desired granularity
    if granularity == "atom":
        pass
    elif granularity == "centroids":
        atoms = convert_structure_to_centroids(atoms)
    else:
        ligand_atoms = filter_dataframe(atoms,
                                        by_column='record_name',
                                        list_of_values=['HETATM'],
                                        boolean=True)
        protein_atoms = filter_dataframe(atoms,
                                        by_column='record_name',
                                        list_of_values=['ATOM'],
                                        boolean=True)
        atoms = subset_structure_to_atom_type(protein_atoms, granularity)
        atoms = pd.concat([atoms, ligand_atoms])

    protein_df = atoms

    # Remove alt_loc residues
    if alt_locs != "include":
        protein_df = remove_alt_locs(protein_df, keep=alt_locs)

    # Remove inserted residues
    if not insertions:
        protein_df = remove_insertions(protein_df)

    # perform chain selection
    protein_df = select_chains(protein_df, chain_selection=chain_selection)

    # Sort dataframe to place HETATMs
    protein_df = sort_dataframe(protein_df)

    if pocket_only:
        if 'HETATM' not in protein_df['record_name'].unique():
            pass
        else:
            retain_residue_ids = []
            dist_mat =compute_distmat(protein_df)
            interacting_nodes = get_interacting_atoms(10, distmat=dist_mat)
            interacting_nodes = list(zip(interacting_nodes[0], interacting_nodes[1]))

            for a1, a2 in interacting_nodes:
                n1 = protein_df.loc[a1, "record_name"]
                n2 = protein_df.loc[a2, "record_name"]
                n1_id = protein_df.loc[a1, "residue_id"]
                n2_id = protein_df.loc[a2, "residue_id"]
                if n1 == 'ATOM' and n2 == 'HETATM':
                    retain_residue_ids.extend([n1_id, n2_id])

            retain_residue_ids = list(set(retain_residue_ids))
            protein_df = filter_dataframe(
                protein_df,
                by_column="residue_id",
                list_of_values=retain_residue_ids,
                boolean=True,
            )

    return protein_df


def label_node_id(
    df: pd.DataFrame,
    granularity: str,
    keep_hets) -> pd.DataFrame:

    df["node_id"] = (
        df["chain_id"].apply(str)
        + ":"
        + df["residue_name"]
        + ":"
        + df["residue_number"].apply(str)
    )

    # Add Alt Loc identifiers
    df["node_id"] = df["node_id"] + ":" + df["alt_loc"].apply(str)
    df["node_id"] = df["node_id"].str.replace(":$", "", regex=True)
    df["residue_id"] = df["node_id"]

    def update_node_id(row):
        node_id = row["node_id"]
        atom_name = row["atom_name"]

        # 检查原始的 node_id 是否包含 het_atm 中的任何元素
        contains_het_atm = any(atom in node_id for atom in keep_hets)

        if contains_het_atm:
            # 如果包含，保留原始的 node_id
            return f"{node_id}:{atom_name}"
        else:
            # 如果不包含，进行拼接更新
            return node_id

    if granularity == "atom":
        df["node_id"] = df["node_id"] + ":" + df["atom_name"]
    else:
        df["node_id"] = df.apply(update_node_id, axis=1)

    return df


def filter_hetatms(
    df: pd.DataFrame, keep_hets: List[str]
) -> List[pd.DataFrame]:

    return [df.loc[df["residue_name"] == hetatm] for hetatm in keep_hets]


def deprotonate_structure(df: pd.DataFrame) -> pd.DataFrame:
    return filter_dataframe(
        df,
        by_column="element_symbol",
        list_of_values=["H", "D", "T"],
        boolean=False,
    )


def remove_alt_locs(
    df: pd.DataFrame, keep: str = "max_occupancy"
) -> pd.DataFrame:
    """
    This function removes alternatively located atoms from PDB DataFrames
    (see https://proteopedia.org/wiki/index.php/Alternate_locations). Among the
    alternative locations the ones with the highest occupancies are left.

    """
    # Sort accordingly
    if keep == "max_occupancy":
        df = df.sort_values("occupancy")
        keep = "last"
    elif keep == "min_occupancy":
        df = df.sort_values("occupancy")
        keep = "first"
    elif keep == "exclude":
        keep = False

    # Filter
    duplicates = df.duplicated(
        subset=["chain_id", "residue_number", "atom_name", "insertion"],
        keep=keep,
    )
    df = df[~duplicates]

    # Unsort
    if keep in ["max_occupancy", "min_occupancy"]:
        df = df.sort_index()

    return df


def remove_insertions(
    df: pd.DataFrame, keep = "first"
) -> pd.DataFrame:

    # Catches unnamed insertions
    duplicates = df.duplicated(
        subset=["chain_id", "residue_number", "atom_name", "alt_loc"],
        keep=keep,
    )
    df = df[~duplicates]

    return filter_dataframe(
        df, by_column="insertion", list_of_values=[""], boolean=True
    )


def convert_structure_to_centroids(df: pd.DataFrame) -> pd.DataFrame:

    centroids = calculate_centroid_positions(df)
    df = df.loc[df["atom_name"] == "CA"].reset_index(drop=True)
    df["x_coord"] = centroids["x_coord"]
    df["y_coord"] = centroids["y_coord"]
    df["z_coord"] = centroids["z_coord"]

    return df


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["chain_id", "residue_number", "atom_number", "insertion"]
        # by = ["atom_number"]
    )


def calculate_centroid_positions(
    atoms: pd.DataFrame
) -> pd.DataFrame:

    centroids = (
        atoms.groupby(
            ["residue_number", "chain_id", "residue_name"]
        )
        .mean()[["x_coord", "y_coord", "z_coord"]]
        .reset_index()
    )

    return centroids


def select_chains(
    protein_df: pd.DataFrame,
    chain_selection: Union[str, List[str]],
) -> pd.DataFrame:

    if chain_selection != "all":
        if isinstance(chain_selection, str):
            raise ValueError(
                "Only 'all' is a valid string for chain selection. Otherwise use a list of strings: e.g. ['A', 'B', 'C']"
            )
        protein_df = filter_dataframe(
            protein_df,
            by_column="chain_id",
            list_of_values=chain_selection,
            boolean=True,
        )

    return protein_df

def construct_pseudoatom_df(xyz_list, b_factor_list=None):
    if b_factor_list is None:
        b_factor_list = [1.00] * len(xyz_list)
    output_df = {'record_name': [],
                 'atom_number': [],
                 'blank_1': [],
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

    for i, (xyz, b_factor) in enumerate(zip(xyz_list, b_factor_list)):
        output_df['record_name'].append('HETATM')
        output_df['atom_number'].append(i + 1)
        output_df['blank_1'].append('')
        output_df['atom_name'].append('PSE')
        output_df['alt_loc'].append('')
        output_df['residue_name'].append('PSE')
        output_df['blank_2'].append('')
        output_df['chain_id'].append('P')
        output_df['residue_number'].append(9999)
        output_df['insertion'].append('')
        output_df['blank_3'].append('')
        output_df['x_coord'].append(xyz[0])
        output_df['y_coord'].append(xyz[1])
        output_df['z_coord'].append(xyz[2])
        output_df['occupancy'].append(1.00)
        output_df['b_factor'].append(b_factor)
        output_df['blank_4'].append('')
        output_df['segment_id'].append('0.0')
        output_df['element_symbol'].append('X')
        output_df['charge'].append(np.nan)
        output_df['line_idx'].append(i)
        output_df['model_id'].append(1)
        output_df['node_id'].append('')
        output_df['residue_id'].append('')
    df = pd.DataFrame(output_df, index=None)
    return df

def subset_structure_to_atom_type(
    df: pd.DataFrame, granularity: str
) -> pd.DataFrame:
    """
    Return a subset of atomic dataframe that contains only certain atom names.

    :param df: Protein Structure dataframe to subset.
    :type df: pd.DataFrame
    :returns: Subset protein structure dataframe.
    :rtype: pd.DataFrame
    """
    return filter_dataframe(
        df, by_column="atom_name", list_of_values=[granularity], boolean=True
    )