import socket
import subprocess
from sys import platform

class TPATH:
    pc_name = socket.gethostname()
    platform = platform

    if platform == "linux" and pc_name == "Mercury":
        ROSETTA_PATH = "/mnt/e/Program_Files/rosetta.source.release-371/"
        SCHRODINGER_PATH = "/mnt/e/Program_Files/Schrodinger_202402/"
        RMSD_SCRIPT = r"E:\\Program_Files\\Schrodinger_202402\\mmshare-v6.6\\python\\common\\rmsd.py"

    elif platform == "linux" and pc_name == "Venus":
        ROSETTA_PATH = "/home/fluoxetine/rosetta.source.release-371/"
        SCHRODINGER_PATH = "/mnt/c/Program\ Files/Schrodinger2024-2/"

    # Rosetta-related
    SIMPLEPEP = ROSETTA_PATH + "main/source/bin/simple_cycpep_predict.mpi.linuxgccrelease"
    SILENT_SPLIT = ROSETTA_PATH + "main/source/bin/extract_pdbs.mpi.linuxgccrelease"

    # Schrodinger-related
    SCHRODINGER_RUN = SCHRODINGER_PATH + "run.exe"
    GLIDE = SCHRODINGER_PATH + "glide.exe"
    LIGPREP = SCHRODINGER_PATH + "ligprep.exe"
    STRUCTCONVERT = SCHRODINGER_PATH + "utilities/structconvert.exe"
    PROTEINPREP = SCHRODINGER_PATH + "utilities/prepwizard.exe"


def convert_wsl_to_windows_path(wsl_path, arg='-w'):
    # 使用wslpath命令将WSL路径转换为Windows路径
    result = subprocess.run(['wslpath', arg, wsl_path], stdout=subprocess.PIPE)
    windows_path = result.stdout.decode('utf-8').strip()
    windows_path = windows_path.replace('\\', '\\\\')
    return windows_path


if __name__ == '__main__':
    wsl_path = '/tmp'
    windows_path = convert_wsl_to_windows_path(wsl_path)
    print(windows_path)


