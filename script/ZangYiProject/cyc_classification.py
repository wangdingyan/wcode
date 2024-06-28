import pandas as pd
# from wcode.protein.seq.embedding import compute_esm_embedding
#
# emb = compute_esm_embedding(['CAAAC', 'CGCGC'], 'sequence')
# print(emb.shape)
#
from wcode.protein.constant import BASE_AMINO_ACIDS
import random
def generate_random_sequence(num):
    output = []
    for i in range(num):
        sequence = []
        sequence.append('C')
        for j in range(7):
            sequence.append(random.choice(BASE_AMINO_ACIDS))
        sequence.append('C')
        sequence = ''.join(sequence)
        output.append(sequence)
    return output

print(generate_random_sequence(10))
