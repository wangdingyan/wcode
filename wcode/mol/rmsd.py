import subprocess
import numpy as np
from numpy import ndarray
from wcode.utils.config import TPATH
from typing import Optional, Any, Tuple
from wcode.protein.biodf import filter_dataframe, read_pdb_to_dataframe


RMSD_SCRIPT = '''{SCHRODINGER_RUN} {RMSD_SCRIPT}'''


def rmsd_pdb(ref_file,
         input_file,
         use_neutral_scaffold=True,
         superimpose=True,
         output_file=None,
         asl=None):
    CMD = RMSD_SCRIPT.format(SCHRODINGER_RUN=TPATH.SCHRODINGER_RUN,
                             RMSD_SCRIPT=TPATH.RMSD_SCRIPT)
    if use_neutral_scaffold:
        CMD += ' -use_neutral_scaffold'
    if superimpose:
        CMD += ' -m'
    CMD += f' {ref_file} {input_file}'
    if output_file is not None:
        CMD += f' -o {output_file}'
    if asl is not None:
        CMD += f' -asl {asl}'

    return_text = subprocess.run(CMD, shell=True, capture_output=True, text=True)
    msg = return_text.stdout

    loc = msg.find("Superimposed RMSD")
    rmsd = float(msg[loc+20: loc+24])
    return rmsd

def rmsd_backbone(ref_file, input_file):
    prt1 = read_pdb_to_dataframe(ref_file)
    prt2 = read_pdb_to_dataframe(input_file)
    xyz1 = filter_dataframe(prt1, by_column='atom_name', list_of_values=['CA'], boolean=True)[['x_coord', 'y_coord', 'z_coord']].to_numpy()
    xyz2 = filter_dataframe(prt2, by_column='atom_name', list_of_values=['CA'], boolean=True)[['x_coord', 'y_coord', 'z_coord']].to_numpy()
    return kabsch_rmsd(xyz1, xyz2, translate=True)


#######################################################################################################################

def rmsd(P: ndarray, Q: ndarray, **kwargs) -> float:
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.

    Parameters
    ----------
    V : array
        (N,D) matrix, where N is points and D is dimension.
    W : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rmsd : float
        Root-mean-square deviation between the two vectors
    """
    diff = P - Q
    return np.sqrt((diff * diff).sum() / P.shape[0])


def kabsch_rmsd(
    P: ndarray,
    Q: ndarray,
    W: Optional[ndarray] = None,
    translate: bool = False,
    **kwargs: Any,
) -> float:
    """
    Rotate matrix P unto Q using Kabsch algorithm and calculate the RMSD.
    An optional vector of weights W may be provided.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    W : array or None
        (N) vector, where N is points.
    translate : bool
        Use centroids to translate vector P and Q unto each other.

    Returns
    -------
    rmsd : float
        root-mean squared deviation
    """

    if translate:
        Q = Q - centroid(Q)
        P = P - centroid(P)

    if W is not None:
        return kabsch_weighted_rmsd(P, Q, W)

    P = kabsch_rotate(P, Q)
    return rmsd(P, Q)


def kabsch_weighted_rmsd(P: ndarray, Q: ndarray, W: Optional[ndarray] = None) -> float:
    """
    Calculate the RMSD between P and Q with optional weights W

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    W : vector
        (N) vector, where N is points

    Returns
    -------
    RMSD : float
    """
    _, _, w_rmsd = kabsch_weighted(P, Q, W)
    return w_rmsd


def kabsch_rotate(P: ndarray, Q: ndarray) -> ndarray:
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    P : array
        (N,D) matrix, where N is points and D is dimension,
        rotated

    """
    U = kabsch(P, Q)

    # Rotate P
    P = np.dot(P, U)
    return P


def kabsch(P: ndarray, Q: ndarray) -> ndarray:
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U: ndarray = np.dot(V, W)

    return U

def kabsch_weighted(
    P: ndarray, Q: ndarray, W: Optional[ndarray] = None
) -> Tuple[ndarray, ndarray, float]:
    """
    Using the Kabsch algorithm with two sets of paired point P and Q.
    Each vector set is represented as an NxD matrix, where D is the
    dimension of the space.
    An optional vector of weights W may be provided.

    Note that this algorithm does not require that P and Q have already
    been overlayed by a centroid translation.

    The function returns the rotation matrix U, translation vector V,
    and RMS deviation between Q and P', where P' is:

        P' = P * U + V

    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    W : array or None
        (N) vector, where N is points.

    Returns
    -------
    U    : matrix
           Rotation matrix (D,D)
    V    : vector
           Translation vector (D)
    RMSD : float
           Root mean squared deviation between P and Q
    """
    # Computation of the weighted covariance matrix
    CMP = np.zeros(3)
    CMQ = np.zeros(3)
    C = np.zeros((3, 3))
    if W is None:
        W = np.ones(len(P)) / len(P)
    W = np.array([W, W, W]).T
    # NOTE UNUSED psq = 0.0
    # NOTE UNUSED qsq = 0.0
    iw = 3.0 / W.sum()
    n = len(P)
    for i in range(3):
        for j in range(n):
            for k in range(3):
                C[i, k] += P[j, i] * Q[j, k] * W[j, i]
    CMP = (P * W).sum(axis=0)
    CMQ = (Q * W).sum(axis=0)
    PSQ = (P * P * W).sum() - (CMP * CMP).sum() * iw
    QSQ = (Q * Q * W).sum() - (CMQ * CMQ).sum() * iw
    C = (C - np.outer(CMP, CMQ) * iw) * iw

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U, translation vector V, and calculate RMSD:
    U = np.dot(V, W)
    msd = (PSQ + QSQ) * iw - 2.0 * S.sum()
    if msd < 0.0:
        msd = 0.0
    rmsd_ = np.sqrt(msd)
    V = np.zeros(3)
    for i in range(3):
        t = (U[i, :] * CMQ).sum()
        V[i] = CMP[i] - t
    V = V * iw
    return U, V, rmsd_



def centroid(X: ndarray) -> ndarray:
    """
    Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.

    https://en.wikipedia.org/wiki/Centroid

    C = sum(X)/len(X)

    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    C : ndarray
        centroid
    """
    C: ndarray = X.mean(axis=0)
    return C

#######################################################################################################################


if __name__ == '__main__':
    # import os
    # import pandas as pd
    #
    # full_names = os.listdir(r'/mnt/c/tmp/2017_Science_Rosetta/5TU6/')
    #
    # names = []
    # rmsds = []
    # for name in full_names:
    #     try:
    #         r = rmsd(#r'D:\\nutshell\\Official\\AIPROJECT\\CycPepModel\\2017_ScienceTest\\10-1_6BEQ.pdb',
    #                   r'C:\\tmp\\ligand_89\\5tu6_5tu6-CP.pdb',
    #                         r'C:\\tmp\\2017_Science_Rosetta\\5TU6\\'+name,
    #                 output_file=r'C:\\tmp\\2017_Science_Rosetta\\5TU6\\align_'+name.replace('.pdb', '.maegz'))
    #         print(name, r)
    #         names.append(name)
    #         rmsds.append(r)
    #     except Exception as e:
    #         print(e)
    #         continue
    #
    # df = pd.DataFrame({'names':names,
    #                    'rmsd': rmsds})
    # df.to_csv(r'/mnt/c/tmp/2017_Science_Rosetta/5TU6.csv', index=False)


    import os
    import pandas as pd

    full_names = os.listdir(r'/mnt/c/tmp/2017_Science_Rosetta/5DI8/')

    names = []
    rmsds = []
    for name in full_names:

        r = rmsd_backbone(#r'D:\\nutshell\\Official\\AIPROJECT\\CycPepModel\\2017_ScienceTest\\10-1_6BEQ.pdb',
                    '/mnt/c/tmp/ligand_89/4zks_4zks-CP.pdb',
                          '/mnt/c/tmp/2017_Science_Rosetta/5DI8/'+name)
        print(name, r)
        names.append(name)
        rmsds.append(r)

    df = pd.DataFrame({'names': names,
                       'rmsd':  rmsds})
    df.to_csv(r'/mnt/c/tmp/2017_Science_Rosetta/5DI8.csv', index=False)









