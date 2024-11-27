from wcode.protein.graph.graph import construct_graph
from wcode.protein.constant import STANDARD_RESI_NAMES
from glob import glob
import pandas as pd

out_dict = {}
for r_name in STANDARD_RESI_NAMES:
    out_dict[r_name] = 0

names = glob('D:\\pdbbind\\v2020-other-PL\\*\\*_pocket.pdb')
# pocket_names = glob('D:\\pdbbind\\v2020-other-PL\\*\\*_pocket.pdb')
for i, n in enumerate(names):

    pdb_id = n.split('\\')[-2]
    print(i, pdb_id)
    try:
        g = construct_graph(n, granularity="CA")
    except:
        continue
    sample_res = g[1]['residue_name'].to_list()

    for r_name in sample_res:
        if r_name in STANDARD_RESI_NAMES:
            out_dict[r_name] += 1
    print("***********POCKET***************")
    for r_name in STANDARD_RESI_NAMES:

        print(f'{r_name}\t{out_dict[r_name]}')
    print("***********POCKETEND***************")

