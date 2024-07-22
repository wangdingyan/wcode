import os
import numpy as np
from wcode.cycpep.fold import extract_pnear


if __name__ == '__main__':
    PATH_11 = '/mnt/c/tmp/validPNEAR/C1/'
    DICT_11 = {}
    names = os.listdir(PATH_11)
    for name in names:
        cls, num = name.split('_')

        if cls not in DICT_11:
            DICT_11[cls] = []
        pnear = extract_pnear(os.path.join(PATH_11, name))
        if pnear is None:
            continue
        DICT_11[cls].append(pnear)

    for n in DICT_11.keys():
        print(f"C1 {n}, Mean: {np.mean(DICT_11[n])} STD: {np.std(DICT_11[n])}, COUNT: {len(DICT_11[n])}")



    PATH_11 = '/mnt/c/tmp/validPNEAR/C2/'
    DICT_11 = {}
    names = os.listdir(PATH_11)
    for name in names:
        cls, num = name.split('_')

        if cls not in DICT_11:
            DICT_11[cls] = []
        pnear = extract_pnear(os.path.join(PATH_11, name))
        if pnear is None:
            continue
        DICT_11[cls].append(pnear)

    for n in DICT_11.keys():
        print(f"C2 {n}, Mean: {np.mean(DICT_11[n])} STD: {np.std(DICT_11[n])}, COUNT: {len(DICT_11[n])}")


    # PATH_11 = '/mnt/c/tmp/validPNEAR/A3/'
    # DICT_11 = {}
    # names = os.listdir(PATH_11)
    # for name in names:
    #     cls, num = name.split('_')
    #
    #     if cls not in DICT_11:
    #         DICT_11[cls] = []
    #     pnear = extract_pnear(os.path.join(PATH_11, name))
    #     if pnear is None:
    #         continue
    #     DICT_11[cls].append(pnear)
    #
    # for n in DICT_11.keys():
    #     print(f"A3, Mean: {np.mean(DICT_11[n])} STD: {np.std(DICT_11[n])}, COUNT: {len(DICT_11[n])}")




