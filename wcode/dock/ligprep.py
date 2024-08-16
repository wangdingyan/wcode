import os
import subprocess
from wcode.utils.config import convert_wsl_to_windows_path, TPATH
from wcode.utils.string import generate_random_string


def ligprep(input_file,
            output_dir,
            retain_ligand=True):
    os.makedirs(output_dir, exist_ok=True)
    input_file = convert_wsl_to_windows_path(input_file)

    outputfile_basename = input_file.replace('.sdf', '_ligprep.sdf').split('\\')[-1]
    cmd = f'{TPATH.LIGPREP} -WAIT -epik -isd {input_file} -osd {outputfile_basename}'

    subprocess.run(cmd, cwd=output_dir, shell=True)


if __name__ == '__main__':
    import sys
    ligprep('/mnt/c/tmp/docking_pipeline_test/imatinib_pubchem.sdf')