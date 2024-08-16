from wcode.protein.graph.graph import construct_graph
from wcode.protein.constant import STANDARD_RESI_NAMES
from glob import glob
import pandas as pd

out_dict = {}
for r_name in STANDARD_RESI_NAMES:
    out_dict[r_name] = []
target_name = []


names = glob('D:\\pdbbind\\v2020-other-PL\\*\\*_protein.pdb')
for i, n in enumerate(names):
    pdb_id = n.split('\\')[-2]
    print(i, pdb_id)
    try:
        g = construct_graph(n)
    except:
        continue
    sample_res = list(set(g[1]['residue_name'].to_list()))

    target_name.append(pdb_id)
    for r_name in STANDARD_RESI_NAMES:
        out_dict[r_name].append(int(r_name in sample_res))

    df = pd.DataFrame({'pdb_id': target_name})
    for r_name in STANDARD_RESI_NAMES:
        df[r_name] = out_dict[r_name]

    df.to_csv('C:\\tmp\\static.csv')

