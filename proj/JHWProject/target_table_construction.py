import os
import numpy as np
import pandas as pd
from glob import glob
from wcode.mol.smilesconverter import SmilesConverter


rootdir = "/cluster/home/wangdingyan/WDrugDataset"
target_table = glob(os.path.join(rootdir, "uniprot", "uniprotkb_reviewed_true_AND_proteome_up*.tsv"))[0]
protein_name_df = pd.read_csv(target_table, sep="\t")
protein_name_list = protein_name_df['Entry'].tolist()
target_list = sorted(protein_name_list)
print(target_list)

activity_table = os.path.join(rootdir, "bind", "BindingDB_All_Activity.tsv")
activity_df = pd.read_csv(activity_table, sep="\t", low_memory=False)
print(activity_df.shape)
print(activity_df.iloc[0])

output_df = pd.DataFrame({t:[] for t in ['canonical_smiles_column']+target_list})
print(output_df)

converter = SmilesConverter()

for i, l in enumerate(activity_df.iterrows()):
    if i % 10000 == 0:
        tmp_df = output_df.dropna(axis=1, how='all')
        tmp_df.to_csv(os.path.join(rootdir, f'output_{i}.csv'))
    uniprot_id = l[1]['UniProt (SwissProt) Primary ID of Target Chain']
    if uniprot_id not in target_list:
        print(f"{i} Target Not Identified, pass.")
        continue
    else:
        smiles = l[1]['Ligand SMILES']
        canonical_smiles = SmilesConverter.canonicalize_smiles(smiles)
        if canonical_smiles is None:
            print(f"{i} SMILES can not be identified, pass.")
            continue
        flag = False
        for name in ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']:
            if pd.isna(l[1][name]):
                continue
            else:
                flag = True
                if canonical_smiles not in output_df.index:
                    output_df.loc[canonical_smiles] = np.nan
                if pd.isna(output_df.at[canonical_smiles, uniprot_id]):
                    output_df.at[canonical_smiles, uniprot_id] = str(l[1][name])
                else:
                    output_df.at[canonical_smiles, uniprot_id] = f"{output_df.at[canonical_smiles, uniprot_id]};{str(l[1][name])}"

        if flag:
            print(f"{i} Recorded.")
        else:
            print(f"{i} No Activity Value can be used.")




