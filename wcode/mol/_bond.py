from rdkit.Chem.rdchem import BondType

########################################################################################################################

def get_bond_features(bond):
    retval = [0, 0, 0, 0, 0]
    bt = RDKIT_BOND_TYPES.get(bond.GetBondType(), 4)
    retval[bt] = 1
    return retval

########################################################################################################################
### Define RDKit BOND TYPES ###
RDKIT_BOND_TYPES = {
    BondType.SINGLE: 0,
    BondType.DOUBLE: 1,
    BondType.TRIPLE: 2,
    BondType.AROMATIC: 3,
    BondType.OTHER: 4,
}