import os
from wcode.protein.graph.graph_conversion import construct_graph, GraphFormatConvertor
import torch
from wcode.pl.merge import merge_protein_ligand_file
from wcode.protein.biodf import read_pdb_to_dataframe, save_pdb_df_to_pdb
from wcode.protein.graph.graph_distance import *

########################################################################################################################

def mark_pocket(protein_path,
                ligand_path,
                distance_threshold=8,
                marked_path=None):

    keep_hets = ['LIG']
    output_path, ligand_smiles = merge_protein_ligand_file(protein_path, ligand_path)
    protein_df = read_pdb_to_dataframe(output_path,
                                       keep_hets=keep_hets,
                                       granularity='atom')

    retain_residue_ids = []
    dist_mat = compute_distmat(protein_df)
    interacting_nodes = get_interacting_atoms(distance_threshold, distmat=dist_mat)
    interacting_nodes = list(zip(interacting_nodes[0], interacting_nodes[1]))

    for a1, a2 in interacting_nodes:
        n1 = protein_df.loc[a1, "record_name"]
        n2 = protein_df.loc[a2, "record_name"]
        n1_id = protein_df.loc[a1, "residue_id"]
        n2_id = protein_df.loc[a2, "residue_id"]
        if n1 == 'ATOM' and n2 == 'HETATM':
            retain_residue_ids.extend([n1_id, n2_id])

    retain_residue_ids = list(set(retain_residue_ids))
    protein_df.loc[protein_df['residue_id'].isin(retain_residue_ids), 'b_factor'] = 1
    save_pdb_df_to_pdb(protein_df, marked_path, hetatms=False)
    return None


def generate_pyg_feature_file(n):
    print(1, n)
    base_dir = "/mnt/c/database/PDBBind/PDBBind_processed"
    protein = os.path.join(base_dir, n, f'{n}_protein_processed.pdb')
    ligand = f'/mnt/c/data/LGDrugAI/ligand_prep/{n}_ligand_prep.sdf'
    mark_pocket(protein,
                ligand,
                marked_path=os.path.join(base_dir, n, f'{n}_pocket_marked.pdb'),
                distance_threshold=5)
    g, df = construct_graph(os.path.join(base_dir, n, f'{n}_pocket_marked.pdb'),
                            granularity='CA',
                            dssp=True,
                            esm=True,
                            pocket_only=False)
    converter = GraphFormatConvertor()
    pyg = converter.convert_nx_to_pyg(g)
    torch.save(pyg, os.path.join('/mnt/c/database/PDBBind/pocket_marked_CA_esm_dssp',
                                 f'{n}_pocket_marked_CA_esm_dssp.pt'))
    with open("/mnt/c/tmp/record.txt", 'a+') as f:
        f.write(f"{n} Success\n")
    # except:
    #     with open("C:\\tmp\\record.txt", 'a+') as f:
    #         f.write(f"{n} Fail\n")

########################################################################################################################


if __name__ == '__main__':
    import os

    from glob import glob
    from wcode.protein.graph.graph_conversion import construct_graph, GraphFormatConvertor
    import torch
    base_dir = '/mnt/c/database/PDBBind/PDBBind_processed'
    names    = os.listdir(base_dir)
    finished_names = os.listdir('/mnt/c/database/PDBBind/pocket_marked_CA_esm_dssp')
    finished_names = [p.split('_')[0] for p in finished_names]
    names = [n for n in names if n not in finished_names]
    failed_protein = ['3vd7']
    names = [n for n in names if n not in failed_protein]
    print(len(names))
    import random
    random.shuffle(names)

    # from multiprocessing import Pool, freeze_support
    # freeze_support()
    # pool = Pool(5)
    for n in names:
        # pool.apply_async(func=generate_pyg_feature_file, args=(n,))
        try:
            generate_pyg_feature_file(n)
        except:
            continue
    # pool.close()
    # pool.join()
    # for n in names:
    #     print(n)
    #     generate_pyg_feature_file(n)

    # pdb_id = '13gs'
    # protein_1 = os.path.join('D:\\PDBBind\\PDBBind_processed', f'{pdb_id}\\{pdb_id}_protein_processed.pdb')
    # ligand_1 = os.path.join('D:\\PDBBind\\PDBBind_processed', f'{pdb_id}\\{pdb_id}_ligand.sdf')
    # mark_pocket(protein_1, ligand_1, marked_path=os.path.join('D:\\PDBBind\\PDBBind_processed', f'{pdb_id}\\{pdb_id}_pocket_marked.pdb'))
    # g, df = construct_graph(os.path.join('D:\\PDBBind\\PDBBind_processed', f'{pdb_id}\\{pdb_id}_pocket_marked.pdb'), pocket_only=False)
    # converter = GraphFormatConvertor()
    # pyg = converter.convert_nx_to_pyg(g)
    # torch.save(pyg, os.path.join('D:\\PDBBind\\PDBBind_processed', pdb_id, f'{pdb_id}_pocket_marked.pt'))
