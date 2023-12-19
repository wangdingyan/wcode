from wcode.protein.graph.graph import merge_protein_ligand_file, read_pdb_to_dataframe, save_pdb_df_to_pdb
from wcode.protein.graph.graph_distance import *
import tempfile

########################################################################################################################
def mark_pocket(protein_path,
                ligand_path,
                distance_threshold=8,
                marked_path=None):

    keep_hets = ['LIG']
    output_path, ligand_smiles = merge_protein_ligand_file(protein_path, ligand_path)
    protein_df = read_pdb_to_dataframe(output_path, keep_hets=keep_hets)

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


########################################################################################################################


if __name__ == '__main__':
    import os
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'sample_data')
    protein_1 = os.path.join(sample_dir, '1a0q_protein_processed.pdb')
    ligand_1 = os.path.join(sample_dir, '1a0q_ligand.sdf')
    mark_pocket(protein_1, ligand_1, marked_path=os.path.join(sample_dir, '1a0q_pocket_marked.pdb'))

