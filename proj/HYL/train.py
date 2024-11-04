import os
from typing import List

from wcode.cycpep.data import Dataset
from wcode.cycpep.fold import fold_pnear, extract_pnear
from wcode.cycpep.utils import generate_random_sequence
from wcode.protein.constant import STANDARD_AMINO_ACID_MAPPING_3_TO_1
from wcode.protein.seq.embedding import compute_esm_embedding, compute_sequence_embeddings_fast
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def makedir(dir):
    os.makedirs(dir, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用GPU


# def amino_acid_one_hot_batch(sequences: List[str]) -> torch.Tensor:
#     # 定义氨基酸一字母表示及其索引
#     amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # 不包括 D 和 L
#     aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}
#
#     max_length = max(len(seq.split()) for seq in sequences)
#     one_hot = torch.zeros(len(sequences), max_length, len(amino_acids) + 2, dtype=torch.float32)
#
#     for i, seq in enumerate(sequences):
#         for j, aa in enumerate(seq.split()):
#             if aa.startswith('D'):
#                 one_hot[i, j, -2] = 1.0
#             else:
#                 one_hot[i, j, -1] = 1.0
#             aa = aa.replace('D', '')
#             one_letter = STANDARD_AMINO_ACID_MAPPING_3_TO_1.get(aa)
#             if one_letter in aa_to_index:
#                 one_hot[i, j, aa_to_index[one_letter]] = 1.0
#
#     return one_hot

def compute_sequence_embeddings(sequences: List[str],
                                batch_size=100) -> torch.Tensor:

    max_length = max(len(s.split()) for s in sequences)
    total_embedding = torch.zeros(len(sequences), max_length, 1282, dtype=torch.float32)

    total_sequence = []
    for i, seq in enumerate(sequences):
        one_letter_sequence = ''
        for j, aa in enumerate(seq.split()):
            if aa.startswith('D'):
                total_embedding[i, j, -2] = 1.0
            else:
                total_embedding[i, j, -1] = 1.0
            aa = aa.replace('D', '')
            one_letter = STANDARD_AMINO_ACID_MAPPING_3_TO_1.get(aa)
            one_letter_sequence += one_letter
        total_sequence.append(one_letter_sequence)

    for i in tqdm(range(0, len(total_sequence), batch_size)):
        batch_sequences = total_sequence[i:i + batch_size]
        sequence_embeddings = compute_esm_embedding(batch_sequences, 'residue')
        sequence_embeddings = torch.tensor(sequence_embeddings, dtype=torch.float32)[:, 1:-1, :]

        for j, embedding in enumerate(sequence_embeddings):
            total_embedding[i + j, :embedding.size(0), :1280] = embedding

    return total_embedding


class TransformerModel(nn.Module):
    def __init__(self, input_size=35, hidden_size=64, output_size=2, num_layers=3, num_heads=4, ff_hidden_size=64):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)  # 输入嵌入层
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size,
                                       nhead=num_heads,
                                       dim_feedforward=ff_hidden_size,
                                       dropout=0.2),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, output_size)

        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入嵌入
        x = self.embedding(x)  # (batch_size, max_seq_length, hidden_size)

        # 转置为 (max_seq_length, batch_size, hidden_size)
        x = x.permute(1, 0, 2)

        # 使用 Transformer 处理序列
        transformer_out = self.transformer_encoder(x)  # (max_seq_length, batch_size, hidden_size)

        # 取最后一个时间步的输出
        last_out = torch.mean(transformer_out, dim=0)  # (batch_size, hidden_size)

        # 通过全连接层进行预测
        output = self.fc(last_out).squeeze()  # (batch_size, output_size)
        return output


def collate_fn(batch):
    seqs = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    labels = torch.Tensor(labels)
    return seqs, labels


def fit(model, dataset):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    dl = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=True, drop_last=False)

    auroc = 0
    i = 0
    while auroc < 0.95:
        i += 1
        predictions = []
        labels = []
        for b in dl:
            model.train()
            optimizer.zero_grad()
            ss, label = b
            input_embedding = dataset.select_embeddings_from_seqs(ss)
            input_embedding = input_embedding.cuda()
            label = (label.cuda() > 0.9).long()

            prediction = model(input_embedding)

            loss = loss_fn(prediction, label)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                prediction = model(input_embedding)
                labels.extend(label.cpu().detach().tolist())
                # predictions.extend(prediction.squeeze().cpu().detach().tolist())
                predictions.extend(prediction.softmax(-1)[:, 1].cpu().detach().tolist())

        auroc = roc_auc_score(labels, predictions)
        print(i, auroc)
    return model


def test(model, dataset, verbose=False):
    with torch.no_grad():
        model.eval()
        dl = DataLoader(dataset, batch_size=10, collate_fn=collate_fn, shuffle=False, drop_last=False)
        predictions = []
        labels = []
        for b in dl:
            ss, label = b
            input_embedding = dataset.select_embeddings_from_seqs(ss)
            input_embedding = input_embedding.cuda()
            label = (label.cuda() > 0.9).long()
            prediction = model(input_embedding)

            labels.extend(label.cpu().detach().tolist())
            predictions.extend(prediction.softmax(-1)[:, 1].cpu().detach().tolist())
        if not verbose:
            return roc_auc_score(labels, predictions)
        else:
            return dataset.seqs, predictions, labels, roc_auc_score(labels, predictions)


if __name__ == '__main__':
    set_seed(42)
    print("一个简单的基于主动学习环肽Pnear训练流程")

    print("1. 设置工作目录")
    PROJECTDIR = '/mnt/d/tmp/CycPepAL_20241028'
    TRAIN_DIR = os.path.join(PROJECTDIR, 'train_dataset')
    TEST_DIR = os.path.join(PROJECTDIR, 'test_dataset')
    MODEL_DIR = os.path.join(PROJECTDIR, 'model_checkpoint')
    LOG_DIR = os.path.join(PROJECTDIR, 'log')
    makedir(PROJECTDIR)
    makedir(TRAIN_DIR)
    makedir(TEST_DIR)
    makedir(MODEL_DIR)
    makedir(LOG_DIR)
    with open(os.path.join(LOG_DIR, f'test_set.tsv'), 'a+') as f:
        f.write('ROUND\tPearsonR\n')

    print("2. 建立测试集")
    test_seqs = generate_random_sequence(22, 7)
    test_seqs.extend(generate_random_sequence(23, 8))
    test_seqs.extend(['DASP DTHR ASN PRO DTHR LYS DASN',
                      'DASP DGLN DSER DGLU PRO DHIS PRO',
                      'DGLN DASP DPRO PRO DLYS THR ASP',
                      'DASP DASP DPRO DTHR PRO ARG DGLN GLN',
                      'DARG GLN DPRO DGLN ARG DGLU PRO GLN'])
    print("测试集序列：")
    print(test_seqs)

    print("3. 生成测试集折叠结果：")
    for seq in test_seqs:
        print(f'folding {seq}')
        fold_pnear(os.path.join(TEST_DIR, seq),
                   mpi_n=8,
                   seq=seq,
                   n_struct=1000,
                   lamd=0.5,
                   frac=1)
    print("测试集Pnear Label")
    for seq in test_seqs:
        print(seq, extract_pnear(os.path.join(TEST_DIR, seq)))

    print("4. 建立模型")
    seq_model = TransformerModel()
    seq_model = seq_model.cuda()

    print("5. 测试一下在测试集上拟合与测试")
    test_dataset = Dataset(TEST_DIR)
    s1 = test(seq_model, test_dataset)
    model = fit(seq_model, test_dataset)
    s2 = test(seq_model, test_dataset)
    print(f"Before Training, PearsonR {s1}")
    print(f"After Training, PearsonR {s2}")

    print("6. 主动学习过程")
    print("***初始化模型***")
    seq_model = TransformerModel()
    torch.save(seq_model.state_dict(), os.path.join(MODEL_DIR, 'round_0.pth'))

    seq_model = seq_model.cuda()
    for round in range(1, 2000):
        print(f'*** 加载现有训练集 ***')
        training_set = Dataset(TRAIN_DIR)
        training_seqs = training_set.seqs

        print(f'*** Round {round} 生成10000个候选序列')
        score_seqs = []
        while len(score_seqs) < 10000:
            if random.random() < 0.5:
                seq = generate_random_sequence(1, 7)[0]
            else:
                seq = generate_random_sequence(1, 8)[0]
            if seq in training_seqs:
                continue
            else:
                score_seqs.append(seq)

        print(f'*** Round {round} 生成10000个候选序列的表征')
        seqs_embeddings = compute_sequence_embeddings_fast(score_seqs)

        print(f'*** Round {round} 使用模型对10000个候选序列进行预测与打分')
        seqs_scores = []
        temp_dl = DataLoader(seqs_embeddings, batch_size=100, shuffle=False, drop_last=False)
        for temp_b in temp_dl:
            seq_model.eval()
            temp_b = temp_b.cuda()
            temp_pred = seq_model(temp_b)
            seqs_scores.extend(temp_pred.softmax(-1)[:, 1].detach().squeeze().tolist())

        print(f'*** Round {round} 挑选前10名进行折叠，生成label')
        top_k = 10
        _, top_indices = torch.topk(torch.Tensor(seqs_scores), top_k)
        top_seqs = [score_seqs[i] for i in top_indices.numpy()]
        top_scores = [seqs_scores[i] for i in top_indices.numpy()]

        for seq in top_seqs:
            print(f'folding {seq}')
            fold_pnear(os.path.join(TRAIN_DIR, seq),
                       mpi_n=8,
                       seq=seq,
                       n_struct=1000,
                       lamd=0.5,
                       frac=1)

        top_labels = [extract_pnear(os.path.join(TRAIN_DIR, s)) for s in top_seqs]
        with open(os.path.join(LOG_DIR, f'round_{round}_train_pred.tsv'), 'a+') as f:
            f.write('SEQ\tPRED\tLABEL\n')
            for i in range(len(top_seqs)):
                f.write(f'{top_seqs[i]}\t{top_scores[i]}\t{top_labels[i]}\n')

        print(f'*** Round {round} 利用训练集更新训练模型')
        seq_model = TransformerModel().cuda()
        train_dataset = Dataset(TRAIN_DIR)
        s1 = test(seq_model, train_dataset)
        seq_model = fit(seq_model, train_dataset)
        s2 = test(seq_model, train_dataset)
        print(f"Before Training, PearsonR {s1}")
        print(f"After Training, PearsonR {s2}")

        torch.save(seq_model.state_dict(), os.path.join(MODEL_DIR, f'round_{round}.pth'))

        print(f'*** Round {round} 对测试集进行测试')
        test_dataset = Dataset(TEST_DIR)
        seqs, preds, labels, s = test(seq_model, test_dataset, verbose=True)
        with open(os.path.join(LOG_DIR, f'test_set.tsv'), 'a+') as f:
            f.write(f'{round}\t{s}\n')

        with open(os.path.join(LOG_DIR, f'round_{round}_test_pred.tsv'), 'a+') as f:
            f.write('SEQ\tPRED\tLABEL\n')
            for i in range(len(seqs)):
                f.write(f'{seqs[i]}\t{preds[i]}\t{labels[i]}\n')
