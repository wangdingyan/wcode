import pandas as pd
from rdkit import Chem
from wcode.mol.fingerprint import tanimoto_similarity

F32880031 = 'C/C=C/C1=CC(=C(C=C1)OCC(CN2CCN(CC2)CC3=CC=CC=C3)O)OC'
F62090486 = 'C1=CC=C(C=C1)C2=NC=CN2CCCNC(=O)NC3=CC=CC(=C3)C(F)(F)F'


df = pd.read_excel('/mnt/c/data/MyNutShell/Official/PAPER/NET/compound2_information.xlsx')
sms = df['Smiles'].tolist()

mol_32880031_ts = [tanimoto_similarity(s1, F32880031) for s1 in sms]
mol_62090486_ts = [tanimoto_similarity(s1, F62090486) for s1 in sms]

df['s1'] = mol_32880031_ts
df['s2'] = mol_62090486_ts

df.to_csv('/mnt/c/data/MyNutShell/Official/PAPER/NET/compound2_information_sml.csv')
