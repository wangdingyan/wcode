from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import numpy as np
from rdkit.DataStructs import FingerprintSimilarity


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


def tanimoto_similarity(smiles1, smiles2):
    # 将SMILES转换为分子对象
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # 生成ECFP4指纹（radius=2, nBits=1024）
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=1024)

    # 计算Tanimoto相似性
    similarity = FingerprintSimilarity(fp1, fp2)

    return similarity

########################################################################################################################
