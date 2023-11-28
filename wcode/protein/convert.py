from biopandas.pdb import PandasPdb
from typing import Any, List, Optional, Union
import os
from pathlib import Path
from wcode.protein.graph.graph_distance import *


# https://github.com/a-r-j/graphein/blob/master/graphein/protein/graphs.py
########################################################################################################################


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





########################################################################################################################


def label_node_id(
    df: pd.DataFrame) -> pd.DataFrame:

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
    df["node_id"] = df["node_id"] + ":" + df["atom_name"]

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


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["chain_id", "residue_number", "atom_number", "insertion"]
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


def convert_structure_to_centroids(df: pd.DataFrame) -> pd.DataFrame:

    centroids = calculate_centroid_positions(df)
    df = df.loc[df["atom_name"] == "CA"].reset_index(drop=True)
    df["x_coord"] = centroids["x_coord"]
    df["y_coord"] = centroids["y_coord"]
    df["z_coord"] = centroids["z_coord"]

    return df


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

    protein_df = label_node_id(protein_df)

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
    if granularity == "centroids":
        atoms = convert_structure_to_centroids(atoms)
    else:
        pass

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
            retain_residue_numbers = []
            dist_mat =compute_distmat(protein_df)
            interacting_nodes = get_interacting_atoms(10, distmat=dist_mat)
            interacting_nodes = list(zip(interacting_nodes[0], interacting_nodes[1]))

            for a1, a2 in interacting_nodes:
                n1 = protein_df.loc[a1, "record_name"]
                n2 = protein_df.loc[a2, "record_name"]
                n1_position = protein_df.loc[a1, "residue_number"]
                n2_position = protein_df.loc[a2, "residue_number"]
                if n1 == 'ATOM' and n2 == 'HETATM':
                    retain_residue_numbers.extend([n1_position, n2_position])

            retain_residue_numbers = list(set(retain_residue_numbers))
            protein_df = filter_dataframe(
                protein_df,
                by_column="residue_number",
                list_of_values=retain_residue_numbers,
                boolean=True,
            )

    return protein_df

if __name__ == '__main__':
    df = read_pdb_to_dataframe('/mnt/d/tmp/5p21.pdb',
                               keep_hets=['GNP'],
                               pocket_only=True)
    dist_mat = compute_distmat(df)
    dist_mat = dist_mat[dist_mat > 10]
    print(np.nan_to_num(dist_mat))
    df.to_excel('/mnt/d/tmp/5p21.xlsx')
    save_pdb_df_to_pdb(df, '/mnt/d/tmp/5p21_2.pdb')