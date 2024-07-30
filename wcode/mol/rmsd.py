import subprocess
from wcode.utils.config import TPATH

RMSD_SCRIPT = '''{SCHRODINGER_RUN} {RMSD_SCRIPT}'''


def rmsd(ref_file,
         input_file,
         use_neutral_scaffold=True,
         superimpose=True,
         output_file=None):
    CMD = RMSD_SCRIPT.format(SCHRODINGER_RUN=TPATH.SCHRODINGER_RUN,
                             RMSD_SCRIPT=TPATH.RMSD_SCRIPT)
    if use_neutral_scaffold:
        CMD += ' -use_neutral_scaffold'
    if superimpose:
        CMD += ' -m'
    CMD += f' {ref_file} {input_file}'
    if output_file is not None:
        CMD += f' -o {output_file}'

    return_text = subprocess.run(CMD, shell=True, capture_output=True, text=True)
    msg = return_text.stdout

    loc = msg.find("Superimposed RMSD")
    rmsd = float(msg[loc+20: loc+24])
    return rmsd


if __name__ == '__main__':
    import os
    import pandas as pd

    full_names = os.listdir(r'/mnt/c/tmp/2017_Science_Maestro/7.2')

    names = []
    rmsds = []
    for name in full_names:
        try:
            r = rmsd(r'D:\\nutshell\\Official\\AIPROJECT\\CycPepModel\\2017_ScienceTest\\7-2_6BEW.pdb',
                            r'C:\\tmp\\2017_Science_Maestro\\7.2\\'+name,
                     output_file=r'C:\\tmp\\2017_Science_Maestro\\7.2\\align_'+name.replace('.pdb', '.maegz'))
            print(name, r)
            names.append(name)
            rmsds.append(r)
        except:
            continue

    df = pd.DataFrame({'names':names,
                       'rmsd': rmsds})
    df.to_csv(r'/mnt/c/tmp/2017_Science_Maestro/7_2.csv', index=False)

