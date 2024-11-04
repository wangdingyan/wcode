import os
import torch
from wcode.protein.seq.embedding import compute_sequence_embeddings_fast
from wcode.cycpep.fold import extract_pnear


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


if __name__ == '__main__':
    dataset = Dataset(dataset_dir='/mnt/d/tmp/crystal')
    for s, l in dataset:
        print(s, l)