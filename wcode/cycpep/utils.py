import random
from wcode.protein.constant import BASE_AMINO_ACIDS, STANDARD_AMINO_ACID_MAPPING_1_TO_3


def generate_random_sequence(num, length=7):
    output = []
    for i in range(num):
        sequence = []
        for j in range(length):
            amino_acid = random.choice(BASE_AMINO_ACIDS)
            sequence.append(amino_acid)
        sequence = ''.join(sequence)
        sequence_3 = []
        for aa_1 in sequence:
            aa_3 = STANDARD_AMINO_ACID_MAPPING_1_TO_3[aa_1]
            if random.random() < 0.25 and aa_3 != 'GLY':  # 1/4的几率
                aa_3 = 'D' + aa_3
            sequence_3.append(aa_3)
        sequence_3 = ' '.join(sequence_3)
        output.append(sequence_3)
    return output


if __name__ == '__main__':
    ls = generate_random_sequence(10)
    print(ls)