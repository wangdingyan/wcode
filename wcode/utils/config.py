import socket
from sys import platform

class TPATH:
    pc_name = socket.gethostname()
    platform = platform

    if platform == "linux" and pc_name == "Mercury":
        SCHRODINGER_RUN = "/mnt/e/Program_Files/Schrodinger_202402/run.exe"
        RMSD_SCRIPT = r"E:\\Program_Files\\Schrodinger_202402\\mmshare-v6.6\\python\\common\\rmsd.py"
        GLIDE = "/mnt/e/Program_Files/Schrodinger_202402/glide.exe"
        SIMPLEPEP = "/mnt/e/Program_Files/rosetta.source.release-371/main/source/bin/simple_cycpep_predict.mpi.linuxgccrelease"
        SILENT_SPLIT = "/mnt/e/Program_Files/rosetta.source.release-371/main/source/bin/extract_pdbs.mpi.linuxgccrelease"
