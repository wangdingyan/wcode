import os
from typing import List
from wcode.cycpep.fold import fold_pnear, extract_pnear
from wcode.cycpep.utils import generate_random_sequence
from wcode.protein.constant import STANDARD_AMINO_ACID_MAPPING_3_TO_1
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr


def makedir(dir):
    os.makedirs(dir, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用GPU


def amino_acid_one_hot_batch(sequences: List[str]) -> torch.Tensor:
    # 定义氨基酸一字母表示及其索引
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # 不包括 D 和 L
    aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

    max_length = max(len(seq.split()) for seq in sequences)
    one_hot = torch.zeros(len(sequences), max_length, len(amino_acids)+2, dtype=torch.float32)

    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq.split()):
            if aa.startswith('D'):
                one_hot[i, j, -2] = 1.0
            else:
                one_hot[i, j, -1] = 1.0
            aa = aa.replace('D', '')
            one_letter = STANDARD_AMINO_ACID_MAPPING_3_TO_1.get(aa)
            if one_letter in aa_to_index:
                one_hot[i, j, aa_to_index[one_letter]] = 1.0

    return one_hot


class RNNModel(nn.Module):
    def __init__(self, input_size=22, hidden_size=64, output_size=1, num_layers=2):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 使用 LSTM 处理序列
        lstm_out, _ = self.lstm(x)  # lstm_out 形状为 (batch_size, max_seq_length, hidden_size)

        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # 通过全连接层进行预测
        output = self.fc(last_out)  # (batch_size, output_size)
        return output


class Dataset():
    def __init__(self,dataset_dir):
        self.dataset_dir = dataset_dir
        self.seqs = os.listdir(self.dataset_dir)

    def __getitem__(self, index):
        seq = self.seqs[index]
        label = extract_pnear(os.path.join(self.dataset_dir, seq))
        return seq, label

    def __len__(self):
        return len(self.seqs)


def collate_fn(batch):
    seqs = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    input_embedding = amino_acid_one_hot_batch(seqs)
    labels = torch.Tensor(labels)
    return input_embedding, labels


def fit(model, dataset):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dl = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)
    for e in range(300):
        for b in dl:
            optimizer.zero_grad()
            input_embedding, labels = b
            input_embedding = input_embedding.cuda()
            labels = labels.cuda()
            prediction = model(input_embedding)
            loss = loss_fn(labels, prediction)
            print(e, loss.detach().cpu().item())
            loss.backward()
            optimizer.step()

    return model


def test(model, dataset):
    dl = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=False, drop_last=False)
    predictions = []
    labels = []
    for b in dl:
        input_embedding, label = b
        input_embedding = input_embedding.cuda()
        label = label.cuda()
        prediction = model(input_embedding).squeeze()

        labels.extend(label.cpu().detach().tolist())
        predictions.extend(prediction.cpu().detach().tolist())

    return round(pearsonr(predictions, labels)[0], 3)


if __name__ == '__main__':
    set_seed(42)
    print("一个简单的基于主动学习环肽Pnear训练流程")

    print("1. 设置工作目录")
    PROJECTDIR = '/mnt/d/tmp/CycPepAL'
    TRAIN_DIR = os.path.join(PROJECTDIR, 'train_dataset')
    TEST_DIR = os.path.join(PROJECTDIR, 'test_dataset')
    makedir(PROJECTDIR)
    makedir(TRAIN_DIR)
    makedir(TEST_DIR)

    print("2. 建立测试集")
    test_seqs = generate_random_sequence(10, 7)
    test_seqs.extend(generate_random_sequence(10, 8))
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
    model = RNNModel()
    model = model.cuda()

    print("5. 测试一下在测试集上拟合与测试")
    test_dataset = Dataset(TEST_DIR)
    s1 = test(model, test_dataset)
    model = fit(model, test_dataset)
    s2 = test(model, test_dataset)
    print(f"Before Training, PearsonR {s1}")
    print(f"After Training, PearsonR {s2}")









