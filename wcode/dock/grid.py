import os
import numpy as np
import tempfile
import subprocess
import shutil
from rdkit import Chem
from wcode.utils.config import TPATH, convert_wsl_to_windows_path

GRID_IN = """
GRID_CENTER {x},{y},{z}
GRIDFILE {pdb}.zip
INNERBOX 15,15,15
OUTERBOX 30,30,30
RECEP_FILE {prot}
"""

CMD = "{GLIDE} -WAIT {infile}"
INFILE = '{pdb}.in'
ZIPFILE = '{pdb}.zip'


def centroid(lig_sdf_file):
    SDMolSupplier = Chem.SDMolSupplier(lig_sdf_file)
    mol_count = len(SDMolSupplier)
    if mol_count != 1:
        raise ValueError("More than 1 molecule contained in this file!")
    mol = next(SDMolSupplier)
    coordinates = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() != 'H':
            coordinates.append(mol.GetConformer().GetPositions()[i])
    return np.mean(coordinates, axis=0)


def make_grid(ligfile,
              pdb_file,
              output_dir):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    x, y, z  = centroid(ligfile)
    pdb_name = os.path.basename(pdb_file).replace('.pdb', '')
    mae_file = convert_wsl_to_windows_path(pdb_file).replace('.pdb', '.mae')
    subprocess.run(f"{TPATH.STRUCTCONVERT} {convert_wsl_to_windows_path(pdb_file)} {mae_file}",
                   cwd=output_dir, shell=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(os.path.join(tmpdirname, 'grid.in'), 'w') as fp:
            fp.write(GRID_IN.format(x=x, y=y, z=z,
                                    pdb=pdb_name,
                                    prot=mae_file))

        cmd = CMD.format(GLIDE=TPATH.GLIDE, infile=convert_wsl_to_windows_path(os.path.join(tmpdirname, 'grid.in')))
        subprocess.run(cmd, cwd=output_dir, shell=True)
        os.remove(pdb_file.replace('.pdb', '.mae'))


if __name__ == '__main__':
    # print(centroid('/mnt/c/tmp/docking_pipeline_test/Structures.sdf'))\
    make_grid('/mnt/c/tmp/docking_pipeline_test/ligand.sdf',
              '/mnt/c/tmp/docking_pipeline_test/receptor.pdb',
              '/mnt/c/tmp/docking_pipeline_test')


