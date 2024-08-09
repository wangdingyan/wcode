import os
from wcode.dock.ligprep import ligprep
from wcode.dock.protprep import protprep
from wcode.dock.grid import make_grid
from wcode.dock.glide import dock
from glob import glob


def autodock(pdb_file,
             native_ligand,
             query_ligand,
             output_dir):

    os.makedirs(output_dir, exist_ok=True)
    if query_ligand.endswith('.sdf'):
        ligprep(query_ligand, output_dir)
    elif query_ligand.endswith('.pdb') or query_ligand.endswith('.mae'):
        protprep(query_ligand, output_dir)
    make_grid(native_ligand, pdb_file, output_dir)
    gird_file = glob(output_dir + '/*.zip')[0]
    try:
        ligand_file = glob(output_dir + '/*ligprep.sdf')[0]
    except:
        ligand_file = glob(output_dir + '/*protprep.mae')[0]
    dock(os.path.join(gird_file), ligand_file, output_dir)


if __name__ == '__main__':
    autodock(pdb_file='/mnt/c/tmp/docking_pipeline_test/protein.pdb',
             native_ligand='/mnt/c/tmp/docking_pipeline_test/native.sdf',
             query_ligand='/mnt/c/tmp/docking_pipeline_test/structures.mae',
             output_dir='/mnt/c/tmp/docking_pipeline_test/output_dir5')

    # import os
    # from glob import glob
    # names = os.listdir('/mnt/c/tmp/target_identification')
    # names = [name.split('_')[0] for name in names]
    # names = list(set(names))
    # for name in names:
    #     if os.path.exists('/mnt/c/tmp/target_identification/' + name):
    #         continue
    #     print(name)
    #     native_ligand = glob('/mnt/c/tmp/target_identification/' + name + '*.sdf')[0]
    #     pdb_file = glob('/mnt/c/tmp/target_identification/' + name + '*.pdb')[0]
    #     query_ligand = '/mnt/c/tmp/target_identification/8PJ7_ZJ9_protein_processed2_8PJ7_ZJ9.sdf'
    #     autodock(pdb_file=pdb_file,
    #              native_ligand=native_ligand,
    #              query_ligand=query_ligand,
    #              output_dir=f'/mnt/c/tmp/target_identification/{name}')



