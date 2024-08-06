import os
import subprocess
from wcode.utils.config import convert_wsl_to_windows_path, TPATH
from wcode.utils.string import generate_random_string


def ligprep(input_file):

    dir_name = os.path.dirname(input_file)
    inputfile_name = os.path.basename(input_file)
    outputfile_name = inputfile_name.replace('.sdf', '_ligprep.sdf')
    cmd = f'{TPATH.LIGPREP} -WAIT -epik -isd {inputfile_name} -osd {outputfile_name}'

    subprocess.run(cmd, cwd=dir_name, shell=True)


if __name__ == '__main__':
    import sys
    ligprep('/mnt/c/tmp/docking_pipeline_test/imatinib_pubchem.sdf')