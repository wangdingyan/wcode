from .fingerprint import smiles2fp
from rdkit import DataStructs
import pandas as pd
from sklearn.cluster import KMeans

########################################################################################################################

def cluster_smiles(smiles,
                   dists=None,
                   n_clusters=50):
    if dists is None:
        fps = []
        for s in smiles:
            fp = smiles2fp(s)
            fps.append(fp)

        dists = []
        nfps = len(fps)
        for i in range(nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
            dists.append(sims)

    mol_dist = pd.DataFrame(dists)
    k_means = KMeans(n_clusters=n_clusters)
    k1 = k_means.fit_predict(mol_dist)
    output = []
    having_clustered_index = []
    assert(len(k1) == len(smiles))
    for k, s in zip(k1, smiles):
        if k in having_clustered_index:
            continue
        else:
            output.append(s)
            having_clustered_index.append(k)
    return output

########################################################################################################################