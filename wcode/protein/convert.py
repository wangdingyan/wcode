import torch_geometric
from wcode.protein.biodf import read_pdb_to_dataframe, save_pdb_df_to_pdb
from wcode.protein.graph.graph import construct_graph, nxg_to_df
from wcode.protein.graph.graph_distance import *
from wcode.protein.graph.graph_conversion import GraphFormatConvertor


# https://github.com/a-r-j/graphein/blob/master/graphein/protein/graphs.py
########################################################################################################################

class ProtConvertor():
    @staticmethod
    def nx2pyg(G):
        nx2pyg_convertor = GraphFormatConvertor()
        return nx2pyg_convertor.convert_nx_to_pyg(G)

    @staticmethod
    def pdb2df(path,
               model_index=1,
               **kwargs):
        df = read_pdb_to_dataframe(path, model_index, **kwargs)
        return df

    @staticmethod
    def df2pdb(df: pd.DataFrame,
               path: str,
               gz: bool = False,
               atoms: bool = True,
               hetatms: bool = True,):
        save_pdb_df_to_pdb(df, path, gz, atoms, hetatms)

    @staticmethod
    def pdb2nx(protein_path,
               ligand_path=None,
               compute_edge_funcs=None,
               keep_hets=[],
               smiles=None,
               pocket_only=False,
               verbose=False):
        return construct_graph(protein_path, ligand_path, compute_edge_funcs, keep_hets, smiles, pocket_only, verbose)

    @staticmethod
    def nx2df(g):
        return nxg_to_df(g)

    @staticmethod
    def pyg2nx(G):
        return torch_geometric.utils.to_networkx(G, node_attrs=['node_id',
                                                                'coords',
                                                                'b_factor',
                                                                'record_name',
                                                                'residue_number',
                                                                'element_symbol'])

########################################################################################################################


if __name__ == '__main__':
    df = read_pdb_to_dataframe('/mnt/d/tmp/5p21.pdb',
                               keep_hets=['GNP'],
                               pocket_only=True)
    dist_mat = compute_distmat(df)
    dist_mat = dist_mat[dist_mat > 10]
    print(np.nan_to_num(dist_mat))
    df.to_excel('/mnt/d/tmp/5p21.xlsx')
    save_pdb_df_to_pdb(df, '/mnt/d/tmp/5p21_2.pdb')