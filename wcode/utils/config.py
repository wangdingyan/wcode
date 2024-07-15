import socket
from sys import platform

class TPATH:
    pc_name = socket.gethostname()
    platform = platform

    if platform == "linux" and pc_name == "Mercury":
        GLIDE = "/mnt/e/Program_Files/Schrodinger_202402/glide.exe"
        SIMPLEPEP = "/mnt/e/Program_Files/rosetta.source.release-371/main/source/bin/simple_cycpep_predict.mpi.linuxgccrelease"

    
