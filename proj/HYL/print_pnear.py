import os
from glob import glob
from wcode.cycpep.fold import extract_pnear
file_names = glob('/mnt/d/tmp/PRCyc/*/*/*/out.silent')
for f_n in file_names:
    seq, fold_num, repeat_num = f_n.split('/')[-4], f_n.split('/')[-3], f_n.split('/')[-2]
    print(seq, fold_num, repeat_num, extract_pnear(os.path.dirname(f_n)))



