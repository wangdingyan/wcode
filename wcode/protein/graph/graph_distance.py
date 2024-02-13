from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np

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