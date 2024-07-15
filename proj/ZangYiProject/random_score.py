import torch
from wcode.protein.seq.embedding import compute_esm_embedding
import numpy as np
import random

# 定义20种常见氨基酸
AMINO_ACIDS = 'CDEFGHIKLMNPQRSTVWYA'

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(1280, 128)
        self.linear2 = torch.nn.Linear(128, 32)
        self.linear3 = torch.nn.Linear(32, 1)
        self.activation = torch.nn.ReLU()
        self.output = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.output(x)
        return x.squeeze()

model = Model()
model.load_state_dict(torch.load('/mnt/c/tmp/model.tdict'))
model.eval()

def score_sequences(seqs):
   seqs_embeddings = compute_esm_embedding(seqs, 'sequence')
   prediction = model(seqs_embeddings)
   return prediction.detach().squeeze().tolist()


def generate_random_peptide():
    return 'C' + ''.join(random.choices(AMINO_ACIDS, k=7)) + 'C'


def main():
    # 设置要保存的文件名
    file_name = '/mnt/c/tmp/peptide_scores.npz'

    # 读取已保存的序列和分数
    try:
        data = np.load(file_name, allow_pickle=True)
        all_sequences = data['sequences'].tolist()
        all_scores = data['scores'].tolist()
    except FileNotFoundError:
        all_sequences = []
        all_scores = []

    for i in range(100000):
        print(i, len(all_sequences))
        new_sequences = []

        # 生成100个唯一的随机序列
        while len(new_sequences) < 100:
            peptide = generate_random_peptide()
            if peptide not in all_sequences:
                new_sequences.append(peptide)
                all_sequences.append(peptide)
            else:
                print(peptide, ' pass')

        # 对新生成的序列打分
        new_scores = score_sequences(new_sequences)

        # 保存新生成的序列和分数
        all_scores.extend(new_scores)

    # 将最终的结果保存到文件
        np.savez(file_name, sequences=all_sequences, scores=all_scores)


if __name__ == '__main__':
    main()