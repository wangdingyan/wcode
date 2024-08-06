import os
import subprocess
import tempfile
from os.path import abspath
from wcode.utils.config import convert_wsl_to_windows_path
from wcode.utils.string import generate_random_string
from wcode.utils.config import TPATH

GLIDE_ES4 = '''GRIDFILE  {grid}
LIGANDFILE   {ligands}
DOCKING_METHOD   confgen
POSES_PER_LIG   100
POSTDOCK_NPOSE   100
PRECISION   SP
NENHANCED_SAMPLING   4
'''

GLIDE = '''GRIDFILE  {grid}
LIGANDFILE   {ligands}
DOCKING_METHOD   confgen
POSES_PER_LIG   30
POSTDOCK_NPOSE   30
PRECISION   SP
'''


def docking_failed(glide_log):
    if not os.path.exists(glide_log):
        return False
    with open(glide_log) as fp:
        logtxt = fp.read()
    phrases = ['** NO ACCEPTABLE LIGAND POSES WERE FOUND **',
               'NO VALID POSES AFTER MINIMIZATION: SKIPPING.',
               'No Ligand Poses were written to external file',
               'GLIDE WARNING: Skipping refinement, etc. because rough-score step failed.']
    return any(phrase in logtxt for phrase in phrases)


def dock(grid_file,
         ligand_file,
         output_dir,
         enhanced=True):

    infile = GLIDE_ES4 if enhanced else GLIDE
    with tempfile.TemporaryDirectory() as tmpdirname:
        random_idx = generate_random_string()
        glide_in = '{}/{}.in'.format(tmpdirname, random_idx)
        with open(glide_in, 'w') as fp:
            fp.write(infile.format(grid=convert_wsl_to_windows_path(abspath(grid_file)),
                                   ligands=convert_wsl_to_windows_path(abspath(ligand_file))))

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        glide_cmd = f'{TPATH.GLIDE} -WAIT -LOCAL -RESTART {convert_wsl_to_windows_path(glide_in)}'
        subprocess.run(glide_cmd, cwd=output_dir, shell=True)


if __name__ == "__main__":
    dock('/mnt/c/tmp/docking_pipeline_test/receptor.zip',
         '/mnt/c/tmp/docking_pipeline_test/imatinib_pubchem_ligprep.sdf',
         '/mnt/c/tmp/docking_pipeline_test/test_20240806')

