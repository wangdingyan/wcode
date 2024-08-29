from rdkit import Chem
from wcode.database.pubchem import get_pubchem_data, smiles_to_pubchem_id
from wcode.database.drugbank import find_drugbank_groups
from wcode.database.pdb import pdb_id_to_pdb_ligand, pdb_ligand_to_pubchem_RAW


import pandas as pd
df = pd.read_csv('C:\\tmp\\KR_missing_list.csv', header=None)
name_list = df[0].to_list()

pdbcode = []
names = []
smileses = []
drugbank_id = []
chembl_id = []
groups = []

for n in name_list:
    try:
        pdb_ligand_ids = pdb_id_to_pdb_ligand(n)
    except:
        print(n, i, 'PDB Ligand IDS Fail')
        continue

    for i in pdb_ligand_ids:
        try:
            pubchem_id = pdb_ligand_to_pubchem_RAW(i)
        except:
            print(n, i, 'URL Fail')
            continue
        try:
            pubchem_information = get_pubchem_data(pubchem_id)
        except:
            print(n, i, 'PubChem ID Fail')
            continue
        try:
            drugbank_groups = find_drugbank_groups(pubchem_information[-2])
        except:
            print(n, i, 'Drugbank ID Fail')
            continue
        name, s, _, c, g = pubchem_information
        pdbcode.append(n+'_'+i)
        names.append(name)
        smileses.append(s)
        drugbank_id.append(g)
        chembl_id.append(c)
        groups.append(drugbank_groups)
        df_out = pd.DataFrame({'pdbcode': pdbcode,
                               'name': names,
                               'smiles': smileses,
                               'drugbank': drugbank_id,
                               'chembl': chembl_id,
                               'group': groups})
        df_out.to_csv('C:\\tmp\\df_out.csv')
        print(n, i, s, c, g, drugbank_groups)

