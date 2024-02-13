from collections import OrderedDict
import torch
from rdkit import Chem
from rdkit.Chem import Atom, Mol
from typing import List, Any, Mapping
from copy import deepcopy
import numpy as np

########################################################################################################################
def get_atom_features(atom: Atom):
    symbol = atom.GetSymbol()
    features = {
        'symbol': symbol,
        'degree': atom.GetTotalDegree(),
        'valence': atom.GetTotalValence(),
        'formal_charge': atom.GetFormalCharge(),
        'num_Hs': atom.GetTotalNumHs(),
        'hybridization': atom.GetHybridization(),
        'aromatic': atom.GetIsAromatic(),  # True of False
        'mass': atom.GetMass() * 0.01,  # scaling
        'EN': EN[symbol] * 0.25,  # scaling
    }
    return _get_sparse(features)

########################################################################################################################



__all__ = ['get_atom_features', 'NUM_ATOM_FEATURES']

ATOM_SYMBOL = ('*', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I')
DEGREE = (0, 1, 2, 3, 4, 5, 6)
VALENCE = (0, 1, 2, 3, 4, 5, 6)
FORMAL_CHARGE = (-1, 0, 1)
NUM_HS = (0, 1, 2, 3, 4)
HYBRIDIZATION = (
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
)
EN = {
    '*': 0.00,
    'C': 2.55,
    'N': 3.04,
    'O': 3.44,
    'F': 3.98,
    'P': 2.19,
    'S': 2.59,
    'Cl': 3.16,
    'Br': 2.96,
    'I': 2.66,
}

FEATURE_INFORM = OrderedDict([
    ['symbol', {'choices': ATOM_SYMBOL, 'allow_unknown': False}],
    ['degree', {'choices': DEGREE, 'allow_unknown': True}],
    ['valence', {'choices': VALENCE, 'allow_unknown': True}],
    ['formal_charge', {'choices': FORMAL_CHARGE, 'allow_unknown': True}],
    ['num_Hs', {'choices': NUM_HS, 'allow_unknown': True}],
    ['hybridization', {'choices': HYBRIDIZATION, 'allow_unknown': True}],
    ['aromatic', {'choices': None}],
    ['mass', {'choices': None}],
    ['EN', {'choices': None}],
])

# fmt: off
ALLOWABLE_ATOM_FEATURES: Mapping[str, List[Any]] = {
    "atomic_num": [1, 6, 7, 8, 9, 12, 14, 15, 16, 17, 19, 30, 35, 53] + ["misc"],  # type: ignore[list-item]
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "num_hs": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "num_radical_e": [0, 1, 2, 3, 4, "misc"],
    "hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}


ALLOWABLE_BOND_FEATURES: Mapping[str, List[Any]] = {
    "bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "stereo": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "is_conjugated": [False, True],
}

for key, val in FEATURE_INFORM.items():
    if val['choices'] is None:
        val['dim'] = 1
    else:
        val['choices'] = {v: i for i, v in enumerate(val['choices'])}
        if val['allow_unknown']:
            val['dim'] = len(val['choices']) + 1
        else:
            val['dim'] = len(val['choices'])

NUM_KEYS = len(FEATURE_INFORM)
NUM_ATOM_FEATURES = sum([val['dim'] for val in FEATURE_INFORM.values()])




def _get_sparse(features: dict) -> list:
    retval = [0] * NUM_ATOM_FEATURES
    idx = 0
    for key, inform in FEATURE_INFORM.items():
        choices, dim = inform['choices'], inform['dim']
        x = features[key]
        if choices is None:
            retval[idx] = x
        elif inform['allow_unknown'] is True:
            retval[idx + choices.get(x, dim - 1)] = 1
        else:
            retval[idx + choices[x]] = 1
        idx += dim
    return retval


def check_dummy_atom(atom) -> bool :
    return atom.GetAtomicNum() == 0


def coords(atom):
    return atom.GetOwningMol().GetConformer(0).GetAtomPosition(atom.GetIdx())


def distance(atom1, atom2):
    return coords(atom1).Distance(coords(atom2))

def angle_atom(atom1, atom2, atom3):
    v1 = coords(atom1) - coords(atom2)
    v3 = coords(atom3) - coords(atom2)
    return v1.AngleTo(v3) * 180.0 / np.pi

def angle_vector(v1, v2):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    angle =  np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    angle *= 180 / np.pi
    if angle > 90:
        angle = 180 - angle
    assert 0 <= angle <= 90, angle
    return angle


def atomname(atom):
    pdb = atom.GetPDBResidueInfo()
    if pdb is None:
        return str(atom.GetIdx())
    return pdb.GetName().strip()


def resname(atom):
    info = atom.GetPDBResidueInfo()
    if info is None:
        return ''
    return ':'.join(map(lambda x: str(x).strip(),
                        [info.GetChainId(), str(info.GetResidueNumber()),
                         info.GetResidueName(), info.GetInsertionCode()]))


def safe_index(allowable_list: List[Any], value: Any) -> int:
    try:
        return allowable_list.index(value)
    except ValueError:
        assert allowable_list[-1] == "misc"
        return len(allowable_list) - 1


def featurize_atom(atom: Chem.Atom):
    return [ALLOWABLE_ATOM_FEATURES["is_aromatic"].index(atom.GetIsAromatic()),
            ALLOWABLE_ATOM_FEATURES["is_in_ring"].index(atom.IsInRing()),
            ALLOWABLE_ATOM_FEATURES["chirality"].index(str(atom.GetChiralTag())),
            safe_index(
                ALLOWABLE_ATOM_FEATURES["hybridization"], str(atom.GetHybridization())
            ),
            safe_index(ALLOWABLE_ATOM_FEATURES["atomic_num"], atom.GetAtomicNum()),
            safe_index(ALLOWABLE_ATOM_FEATURES["degree"], atom.GetTotalDegree()),
            safe_index(
                ALLOWABLE_ATOM_FEATURES["formal_charge"], atom.GetFormalCharge()
            ),
            safe_index(ALLOWABLE_ATOM_FEATURES["num_hs"], atom.GetTotalNumHs()),
            safe_index(
                ALLOWABLE_ATOM_FEATURES["num_radical_e"], atom.GetNumRadicalElectrons()
            ),
            np.array(coords(atom))[0],
            np.array(coords(atom))[1],
            np.array(coords(atom))[2]]


def featurize_atom_one_hot(atom):
    output = np.concatenate([
        one_of_k_encoding_unk(ALLOWABLE_ATOM_FEATURES["is_aromatic"])(atom.GetIsAromatic()),
        one_of_k_encoding_unk(ALLOWABLE_ATOM_FEATURES["is_in_ring"])(atom.IsInRing()),
        one_of_k_encoding_unk(ALLOWABLE_ATOM_FEATURES["chirality"])(str(atom.GetChiralTag())),
        one_of_k_encoding_unk(ALLOWABLE_ATOM_FEATURES["hybridization"])(str(atom.GetHybridization())),
        one_of_k_encoding_unk(ALLOWABLE_ATOM_FEATURES["atomic_num"])(atom.GetAtomicNum()),
        one_of_k_encoding_unk(ALLOWABLE_ATOM_FEATURES["degree"])(atom.GetTotalDegree()),
        one_of_k_encoding_unk(ALLOWABLE_ATOM_FEATURES["formal_charge"])(atom.GetFormalCharge()),
        one_of_k_encoding_unk(ALLOWABLE_ATOM_FEATURES["num_hs"])(atom.GetTotalNumHs()),
        one_of_k_encoding_unk(ALLOWABLE_ATOM_FEATURES["num_radical_e"])(atom.GetNumRadicalElectrons()),
    ])

    return output

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

if __name__ == '__main__':
    mol = Chem.MolFromSmiles('COCCC')
    atom = list(mol.GetAtoms())[0]
    print(get_atom_features(atom))

