import os
from glob import glob
import torch_geometric
from wcode.protein.biodf import read_pdb_to_dataframe, save_pdb_df_to_pdb
from wcode.protein.graph.graph import construct_graph, nxg_to_df
from wcode.protein.convert import ProtConvertor
from multiprocessing import Pool
from multiprocessing import freeze_support

import os
from glob import glob

class RRedockDataset():
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def get_case_names(self):
        return os.listdir(self.base_dir)

    def get_sample_names(self, case_name):
        return os.listdir(os.path.join(self.base_dir, case_name))

    def get_sample_files_paths(self, case_name, sample_name):
        return glob(os.path.join(self.base_dir, case_name, sample_name, '*'))

    def get_all_sample_files_paths(self):
        all_file_paths = []
        for cn in self.get_case_names():
            for sample in self.get_sample_names(cn):
                fs = self.get_sample_files_paths(cn, sample)
                fs = [f for f in fs if 'merge' not in f]
                fs = [f for f in fs if not f.endswith('.pt')]
                f1, f2 = fs

                if f1.endswith('pdb'):
                    pdb_file, ligand_file = f1, f2
                else:
                    pdb_file, ligand_file = f2, f1

                all_file_paths.append((pdb_file, ligand_file))

        return all_file_paths


rredockdataset = RRedockDataset('D:\\pdb_sdf_dataset_test')
all_file_names = rredockdataset.get_all_sample_files_paths()

from wcode.protein.graph.graph import construct_graph
from wcode.protein.convert import ProtConvertor
import torch


def generate_pyg_file(name_tuple):
    pdb_file, ligand_file = name_tuple
    g, df  = construct_graph(protein_path=pdb_file, ligand_path=ligand_file, pocket_only=True)
    pyg = ProtConvertor.nx2pyg(g)
    torch.save(pyg, ligand_file.replace('.sdf', '.pt'))

if __name__ == '__main__':
    freeze_support()
    pool = Pool(8)
    for name in all_file_names:
        pool.apply_async(func=generate_pyg_file, args=(name,))
    pool.close()
    pool.join()