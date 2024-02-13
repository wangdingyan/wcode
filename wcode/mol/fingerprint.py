from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import numpy as np

########################################################################################################################

def smiles2fp(smiles,
              fp_type=None,
              as_np=True):
    """

    :param smiles:
    :param fp_type:
    :return:
    """
    if fp_type is None:
        fp_type = ['morgan']
    fps = {fpt: [] for fpt in fp_type}
    for i in range(len(smiles)):
        rdkit_mol = Chem.MolFromSmiles(smiles[i])

        for fpt in fp_type:
            if fpt == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, 2, 1024)
            elif fpt == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(rdkit_mol)

            fps[fpt].append(fp)
    if as_np:
        return [np.array(fps[fpt], dtype=np.int64) for fpt in fp_type]
    else:
        return [fps[fpt] for fpt in fp_type]

########################################################################################################################
