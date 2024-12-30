import os
import torch
# from wcode.cycpep.fold import extract_pnear


class Dataset():
    def __init__(self, dataset_dir):
        self.embedding_dict = None
        self.dataset_dir = dataset_dir
        self.seqs = os.listdir(self.dataset_dir)
        print("Computing Dataset Embeddings...")
        # self.compute_seqs_embeddings()

    def __getitem__(self, index):
        s = self.seqs[index]
        label = extract_pnear(os.path.join(self.dataset_dir, s))
        return s, label

    def __len__(self):
        return len(self.seqs)

    def compute_seqs_embeddings(self):
        embeddings = compute_sequence_embeddings_fast(self.seqs)
        self.embedding_dict = {s: emb for s, emb in zip(self.seqs, embeddings)}

    def select_embeddings_from_seqs(self, test_sequences):
        embeddings = [self.embedding_dict[s] for s in test_sequences]
        embeddings = torch.stack(embeddings)
        return embeddings


def generate_cyclic_permutations(peptide, num_augments):


    n = len(peptide)
    if n <= 1:
        raise ValueError("环肽序列的长度必须大于1")

    if num_augments >= n:
        raise ValueError(f"增强数量不能大于或等于环肽序列的长度 - 1, 当前最大值为 {n - 1}")

    # augmentations = set()
    augmentations = []

    for i in range(1, n):

        rotated_peptide = peptide[i:] + peptide[:i]
        augmentations.append(rotated_peptide)

        if len(augmentations) == num_augments:
            break

    # 返回增强序列列表
    return augmentations

if __name__ == '__main__':
    # dataset = Dataset(dataset_dir='/mnt/d/tmp/crystal')
    # for s, l in dataset:
    #     print(s, l)
    peptide = "ABCDEFG"
    # peptide = ['LYS', 'ASP', 'ALA', 'HIS']
    num_augments = 3
    augmented_sequences = generate_cyclic_permutations(peptide, num_augments)

    print(f"输入环肽序列: {peptide}")
    print(f"生成的增强序列: {augmented_sequences}")