import numpy as np

def pnear_calculation(scores,
                      RMSDs,
                      lam=0.5,
                      k_BT=1):
    scores = scores - min(scores)

    pnear_a = 0
    pnear_b = 0

    for score, RMSD in zip(scores, RMSDs):
        pnear_a += np.exp(-(RMSD ^ 2) / (lam ^ 2)) * np.exp(- score/k_BT)
        pnear_b += np.exp(-score/k_BT)

    return pnear_a / pnear_b




