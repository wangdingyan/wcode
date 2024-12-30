import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from wcode.protein.seq.embedding import compute_esm_embedding
from sklearn.metrics import roc_auc_score
import numpy as np

class CustomDataset(Dataset):
    def __init__(self,
                 df,
                 transform=None,
                 mode='Train'):
        self.df = df
        self.transform = transform
        self.length = len(self.df)
        self.mode = mode

    def __getitem__(self, index):
        item = self.df[index]
        if self.transform:
            item = self.transform(item)

        return item[0], item[1], item[2]

    def __len__(self):
        return self.length


if __name__ == '__main__':

    train_df = pd.read_csv('/home/wang_ding_yan/data/wjz_trans/wjz_training.csv')
    train_items = list(train_df.iterrows())
    train_sequences = [s[1]['sequence'] for s in train_items]
    train_label = [s[1]['label'] for s in train_items]
    train_dataset = [[s, int(l>0)] for s, l in zip(train_sequences, train_label)]

    test_df = pd.read_csv('/home/wang_ding_yan/data/wjz_trans/wjz_test.csv')
    test_items = list(test_df.iterrows())
    test_sequences = [s[1]['sequence'] for s in test_items]
    test_label = [s[1]['label'] for s in test_items]
    test_dataset = [[s, int(l>0)] for s, l in zip(test_sequences, test_label)]

    training_sequences = [item[0] for item in train_dataset]
    test_sequences = [item[0] for item in test_dataset]

    training_embeddings = compute_esm_embedding(training_sequences, 'sequence')
    test_embeddings = compute_esm_embedding(test_sequences, 'sequence')

    for item, emb in zip(train_dataset, training_embeddings):
        item.append(emb)

    for item, emb in zip(test_dataset, test_embeddings):
        item.append(emb)

training_dataset = CustomDataset(train_dataset)
test_dataset = CustomDataset(test_dataset)

def collate_fn(batch):
    seqs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    seq_embeddings = [item[2] for item in batch]

    # seq_embedding = compute_esm_embedding(seqs, 'sequence')
    # embeddings = torch.stack(embeddings)
    return torch.stack(seq_embeddings), torch.Tensor(labels)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(1280, 128)
        self.linear2 = torch.nn.Linear(128, 32)
        self.linear3 = torch.nn.Linear(32, 1)
        self.activation = torch.nn.ReLU()
        self.output = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(0.3)

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
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_loader = torch.utils.data.DataLoader(training_dataset, collate_fn=collate_fn, batch_size=20, shuffle=True, drop_last=True)
train_noshuffle_loader = torch.utils.data.DataLoader(training_dataset, collate_fn=collate_fn, batch_size=20, shuffle=False, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=20, shuffle=False)

for epoch in range(1000):
    model = model.eval()

    # test set evaluation
    test_predictions = []
    test_labels = []

    for batch in test_loader:
        predictions = model(batch[0])
        labels = batch[1]
        test_predictions.extend(predictions.detach().numpy().tolist())
        test_labels.extend(labels.detach().numpy().tolist())

    test_loss = loss_fn(torch.Tensor(test_predictions), torch.Tensor(test_labels))
    test_metrics = roc_auc_score(np.array(test_labels), np.array(test_predictions))
    print(f"Test Loss {test_loss}, Test Metrics {test_metrics}")

    df = pd.DataFrame({'test_sequences': test_sequences,
                       'test_labels': test_labels,
                       'test_predictions': test_predictions})
    df.to_csv('/home/wang_ding_yan/data/wjz_trans/test_set_prediction.csv')

    train_predictions = []
    train_labels = []
    for batch in train_noshuffle_loader:
        predictions = model(batch[0])
        labels = batch[1]
        train_predictions.extend(predictions.detach().numpy().tolist())
        train_labels.extend(labels.detach().numpy().tolist())

    train_loss = loss_fn(torch.Tensor(train_predictions), torch.Tensor(train_labels))
    train_metrics = roc_auc_score(np.array(train_labels), np.array(train_predictions))
    print(f"Train Loss {train_loss}, Train Metrics {train_metrics}")

    df = pd.DataFrame({'train_sequences': training_sequences,
                       'train_labels': train_labels,
                       'train_predictions': train_predictions})
    df.to_csv('/home/wang_ding_yan/data/wjz_trans/train_set_prediction.csv')

    model = model.train()
    model.zero_grad()

    for batch in train_loader:
        predictions = model(batch[0])
        labels = batch[1]
        weight = []
        for l in labels:
            if l == 1:
                weight.append(100)
            else:
                weight.append(1)
        weight = torch.Tensor(weight)
        # loss_fn2 = nn.BCELoss(weight=weight)
        loss_fn2 = nn.BCELoss()
        loss = loss_fn2(predictions, labels)
        # print(loss.item())
        loss.backward()
        optimizer.step()
        model.zero_grad()

torch.save(model.state_dict(), '/home/wang_ding_yan/data/wjz_trans/model_save.pt')
model = model.eval()

while True:
    import random

    chars = ['Y', 'F', 'W', 'H', 'C', 'S', 'T', 'L', 'I', 'V', 'A', 'R', 'K', 'H']
    sequences = []
    for _ in range(100):
        length = random.choice([8])  # 随机选择序列的长度
        sequence = [random.choice(chars) for _ in range(length - 2)]  # 前length-1个字符
        if length == 8:  # 如果序列长度为8，最后一位可以是C
            # sequence.append('C')
            sequence.append('RC')
        else:
            sequence.append(random.choice(chars))  # 如果序列长度为7，最后一位不能是C
        sequences.append(''.join(sequence))

    output_embeddings = compute_esm_embedding(sequences, 'sequence')
    output_dataset = [[s, 0, e] for s, e in zip(sequences, output_embeddings)]
    output_dataset = CustomDataset(output_dataset)
    output_loader = torch.utils.data.DataLoader(output_dataset, collate_fn=collate_fn, batch_size=16, shuffle=False)

    output_predictions = []
    output_labels = []
    for batch in output_loader:
        predictions = model(batch[0])
        labels = batch[1]
        output_predictions.extend(predictions.detach().numpy().tolist())
        output_labels.extend(labels.detach().numpy().tolist())

    with open('/home/wang_ding_yan/data/wjz_trans/wjz_output.csv', 'a+') as f:
        for s, p in zip(sequences, output_predictions):
            f.write(f'{s},{p}\n')







