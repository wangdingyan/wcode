from wcode.cycpep.utils import generate_random_sequence
from wcode.cycpep.fold import fold_pnear, extract_pnear
from random import choice

seqs = ["DILE MET PHE VAL GLU ALA THR",
       "HIS LYS TRP ASN PRO DLEU ARG",
       "GLU LYS ASP PRO ILE ILE MET",
       "LEU SER GLY VAL LYS VAL GLN",
       "MET ILE TRP TRP GLN LEU THR",
       "ASP ALA VAL ARG CYS GLY ARG",
       "ASP VAL VAL TYR HIS THR LEU",
       "DILE PHE DGLU DMET MET ASP DHIS"]

for seq in seqs:
    for num in [1000, 5000, 10000]:
        for t in range(2):
            print(num, seq, t)
            fold_pnear(f'/mnt/d/tmp/pnear_repeat/{seq}_{num}_{t}',
                       mpi_n=8,
                       seq=seq,
                       n_struct=num,
                       lamd=0.5,
                       frac=1)
