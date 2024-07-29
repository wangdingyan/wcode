import os
import sys
import torch

from wcode.protein.graph.graph_conversion import construct_graph, GraphFormatConvertor
from wcode.pl.pocket import mark_pocket


def generate_pyg_feature_file(n):
    basedir = "D:\\database\\PDBBind"

    protein_dir = os.path.join(basedir, "protein_processed")
    protein = os.path.join(protein_dir, f'{n}_protein_processed.pdb')
    ligand = os.path.join(basedir, f'ligand_prep/{n}_ligand_prep.sdf')
    mark_pocket(protein,
                ligand,
                marked_path=os.path.join(protein_dir+'_pocket_marked', f'{n}_pocket_marked.pdb'),
                distance_threshold=6)
    g, df = construct_graph(os.path.join(protein_dir+'_pocket_marked', f'{n}_pocket_marked.pdb'),
                            granularity='atom')
    converter = GraphFormatConvertor()
    pyg = converter.convert_nx_to_pyg(g)
    torch.save(pyg, os.path.join(basedir, 'pyg_feature',
                                 f'{n}_pocket_marked.pt'))


if __name__ == '__main__':
    ligand_dir = 'D:\\database\\PDBBind\\ligand_prep'
    names      = os.listdir(ligand_dir)
    names      = [p.split('_')[0] for p in names]

    finished_names = os.listdir('D:\\database\\PDBBind\\protein_processed_pocket_marked')
    finished_names = [p.split('_')[0] for p in finished_names]
    names = [n for n in names if n not in finished_names]

    print(len(names))
    for i, n in enumerate(names):
        try:
            generate_pyg_feature_file(n)
            print(f'{i} {n} Success.')
        except:
            print(f'{i} {n} Failed')
            continue



