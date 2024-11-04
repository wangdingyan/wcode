from wcode.cycpep.utils import generate_random_sequence
from wcode.cycpep.fold import fold_pnear, extract_pnear
from random import choice

while True:
    length = choice([6, 8, 10])
    if length == 6:
        lbd = 0.5
    elif length == 8:
        lbd = 0.5
    elif length == 10:
        lbd = 1
    seq = seq_lists = generate_random_sequence(1, length)[0]
    for num in [1000, 5000, 10000]:
        for t in range(2):
            print(num, seq, t)
            fold_pnear(f'/mnt/d/tmp/pnear_repeat/{seq}_{num}_{t}',
                       mpi_n=8,
                       seq=seq,
                       n_struct=num,
                       lamd=lbd,
                       frac=1)
