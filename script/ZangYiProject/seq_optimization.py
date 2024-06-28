from copy import deepcopy
from wcode.protein.constant import BASE_AMINO_ACIDS


seq_for_optimization = ['CESRVAYYC', 'CTESQGSLC', 'CQEGVLARC']
optional_seqs = []
for seq in seq_for_optimization:
    for i in range(1, 8):
        for symbol in BASE_AMINO_ACIDS:
            list_of_seq = deepcopy(list(seq))
            list_of_seq[i] = symbol
            list_of_seq = ''.join(list_of_seq)
            if list_of_seq not in optional_seqs:
                optional_seqs.append(list_of_seq)

print(len(optional_seqs))