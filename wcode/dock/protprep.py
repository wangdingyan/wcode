import os
import subprocess
from wcode.utils.config import convert_wsl_to_windows_path, TPATH


def protprep(input_file,
             output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_file = convert_wsl_to_windows_path(input_file)
    if input_file.endswith('.pdb'):
        outputfile_basename = input_file.replace('.pdb', '_protprep.mae').split('\\')[-1]
    elif input_file.endswith('.mae'):
        outputfile_basename = input_file.replace('.mae', '_protprep.mae').split('\\')[-1]
    cmd = f'{TPATH.PROTEINPREP}  {input_file} {outputfile_basename} -fillsidechains -disulfides -assign_all_residues -rehtreat -max_states 4 -epik_pH 7.4 -epik_pHt 2.0 -antibody_cdr_scheme Kabat -samplewater -propka_pH 7.4 -fix -f S-OPLS -rmsd 0.3 -watdist 5.0 -WAIT -HOST localhost:8'

    subprocess.run(cmd, cwd=output_dir, shell=True)


if __name__ == '__main__':
    protprep('/mnt/c/tmp/docking_pipeline_test/ligand.mae', '/mnt/c/tmp/docking_pipeline_test/test')