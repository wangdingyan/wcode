import random
from wcode.protein.constant import BASE_AMINO_ACIDS, STANDARD_AMINO_ACID_MAPPING_1_TO_3


def generate_random_sequence(num):
    output = []
    for i in range(num):
        sequence = []
        for j in range(7):
            sequence.append(random.choice(BASE_AMINO_ACIDS))
        sequence = ''.join(sequence)
        sequence_3 = [STANDARD_AMINO_ACID_MAPPING_1_TO_3[a] for a in sequence]
        sequence_3 = ' '.join(sequence_3)
        output.append(sequence_3)
    return output


if __name__ == '__main__':
    ls = generate_random_sequence(10)
    print(ls)