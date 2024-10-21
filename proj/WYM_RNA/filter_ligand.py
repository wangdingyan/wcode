import pandas as p
import gzip
from rdkit import Chem
from rdkit.Chem import QED

with open('/home/fluoxetine/tmp/output.txt', 'r') as f:
    lines = f.readlines()

pdb_list = []
for l in lines:
    ele = l.split('.')[0][-4:]
    pdb_list.append(ele)
pdb_list = list(set(pdb_list))
print(pdb_list)
print(len(pdb_list))

from biopandas.mmcif import PandasMmcif
pmmcif = PandasMmcif()

smiles_dict = {}
with gzip.open('/mnt/d/WDrugDataset/data/ligand.tsv.gz', 'rt') as f:
    # 逐行读取
    for line in f:
        code = line.split('\t')[0]
        smiles = line.split('\t')[4].split(';')[0]
        smiles_dict[code] = smiles

for pdb in pdb_list:
    cif_file = f'/mnt/d/WDrugDataset/pdb/data/structures/divided/mmCIF/{pdb[1:3]}/{pdb}.cif.gz'
    df = pmmcif.read_mmcif(cif_file)
    ligand_codes = set(df.df['HETATM']['label_comp_id'].tolist())
    for l in ligand_codes:
        if l not in smiles_dict:
            continue
        else:
            smiles = smiles_dict[l]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            qed_score = QED.qed(mol)
            print(f'{pdb}\t{smiles}\t{qed_score}')
            with open('/mnt/c/tmp/ligand_score.tsv', 'a+') as fw:
                fw.write(f'{pdb}\t{smiles}\t{qed_score}\n')

