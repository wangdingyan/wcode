import pandas as pd
from torch.utils.data import Dataset
import random
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from wcode.protein.seq.embedding import compute_esm_embedding
from wcode.protein.constant import BASE_AMINO_ACIDS
from tqdm import tqdm

df = pd.read_excel('/mnt/d/nutshell/Official/GG/A-ZangYiProject/100_10_1uM.xlsx')
second_items = list(df.iterrows())
second_items_sequences = [s[1]['sequence'] for s in second_items]

df = pd.read_excel('/mnt/d/nutshell/Official/GG/A-ZangYiProject/GIPR.xlsx')
first_items = list(df.iterrows())
first_items_seqences = [s[1]['seq'] for s in first_items if s[1]['seq'] not in second_items_sequences]
first_items = []
for i, s in enumerate(first_items_seqences):
    first_items.append((i + len(second_items), {'sequence': s,
                                                'GIPR_1': 0,
                                                'GLP1R_1': 0}))

second_items.extend(first_items)
random.shuffle(second_items)

training_dataset = []
test_dataset = []

df = pd.read_csv('/mnt/d/nutshell/Official/GG/A-ZangYiProject/round2_seq.csv')
round2_items = list(df.iterrows())
round2_items_sequences = [s[1]['Seq'] for s in round2_items]
round2_items = []
for i, s in enumerate(round2_items_sequences):
    round2_items.append([s,  1])

for i, item in tqdm(enumerate(second_items)):
    if item[1]['sequence'] in ['CESRVAYYC',
                               'CTESQGSLC',
                               'CQEGVLARC',
                               'CHIRAHNSC',
                               'CDATIHYSC',
                               'CSGLDSGHC']:
        sequence = item[1]['sequence']
        test_dataset.append([item[1]['sequence'],
                             int(item[1]['GIPR_1'] > 0)])
        continue

    else:
        sequence = item[1]['sequence']
        training_dataset.append([item[1]['sequence'],
                                 int(item[1]['GIPR_1'] > 0)])




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


additional_train = generate_random_sequence(2000)
for s in tqdm(additional_train):
    training_dataset.append([s, 0])

addition_test = generate_random_sequence(1000)
for s in tqdm(addition_test):
    if s not in additional_train:
        test_dataset.append([s, 0])

addition_test_2 = generate_random_sequence(2000)
for s in tqdm(addition_test_2):
    if s not in additional_train:
        round2_items.append([s, 0])

test_dataset_round2 = round2_items

for i, s in enumerate(training_dataset):
    print(i, s)

for i, s in enumerate(test_dataset):
    print(i, s)

training_sequences = [item[0] for item in training_dataset]
test_sequences = [item[0] for item in test_dataset]
test_sequences_round2 = [item[0] for item in test_dataset_round2]
training_embeddings = compute_esm_embedding(training_sequences, 'sequence')
test_embeddings = compute_esm_embedding(test_sequences, 'sequence')
test_embeddings_round2 = compute_esm_embedding(test_sequences_round2, 'sequence')

for item, emb in zip(training_dataset, training_embeddings):
    item.append(emb)

for item, emb in zip(test_dataset, test_embeddings):
    item.append(emb)

for item, emb in zip(test_dataset_round2, test_embeddings_round2):
    item.append(emb)


class CustomDataset(Dataset):
    def __init__(self,
                 df,
                 transform=None,
                 mode='Train'):
        self.df = df
        self.transform = transform
        self.len = len(self.df)
        self.mode = mode

    def __getitem__(self, index):
        item = self.df[index]
        if self.transform:
            item = self.transform(item)

        return item[0], item[1], item[2]

    def __len__(self):
        return self.len


training_dataset = CustomDataset(training_dataset)
test_dataset = CustomDataset(test_dataset)
test_dataset_round2 = CustomDataset(test_dataset_round2)


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
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_loader = torch.utils.data.DataLoader(training_dataset, collate_fn=collate_fn, batch_size=16, shuffle=True, drop_last=True)
train_noshuffle_loader = torch.utils.data.DataLoader(training_dataset, collate_fn=collate_fn, batch_size=16, shuffle=False, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=16, shuffle=False)
test_loader_round_2 = torch.utils.data.DataLoader(test_dataset_round2, collate_fn=collate_fn, batch_size=16, shuffle=False)

for epoch in range(500):
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
    df.to_csv('/mnt/c/tmp/test_set_prediction.csv')

    # test set evaluation
    test_predictions = []
    test_labels = []

    for batch in test_loader_round_2:
        predictions = model(batch[0])
        labels = batch[1]
        test_predictions.extend(predictions.detach().numpy().tolist())
        test_labels.extend(labels.detach().numpy().tolist())

    test_loss = loss_fn(torch.Tensor(test_predictions), torch.Tensor(test_labels))
    test_metrics = roc_auc_score(np.array(test_labels), np.array(test_predictions))
    print(f"Test Loss Round 2 {test_loss}, Test Metrics Round 2 {test_metrics}")

    df = pd.DataFrame({'test_sequences': test_sequences_round2,
                       'test_labels': test_labels,
                       'test_predictions': test_predictions})
    df.to_csv('/mnt/c/tmp/test_set_prediction_round2.csv')


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
    df.to_csv('/mnt/c/tmp/train_set_prediction.csv')

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
        loss_fn2 = nn.BCELoss(weight=weight)
        loss = loss_fn2(predictions, labels)
        # print(loss.item())
        loss.backward()
        optimizer.step()
        model.zero_grad()


from copy import deepcopy
seq_for_optimization = ['CESRVAYYC']
optional_seqs = []
for seq in seq_for_optimization:
    for i in range(1, 8):
        for symbol in BASE_AMINO_ACIDS:
            list_of_seq = deepcopy(list(seq))
            list_of_seq[i] = symbol
            list_of_seq = ''.join(list_of_seq)
            if list_of_seq not in optional_seqs:
                optional_seqs.append(list_of_seq)

model = model.eval()
option_seqs_embeddings = compute_esm_embedding(optional_seqs, 'sequence')
prediction = model(option_seqs_embeddings)
output_df = pd.DataFrame({'seq': optional_seqs,
                          'prediction': prediction.detach().squeeze().numpy().tolist()})
output_df.to_csv('/mnt/c/tmp/CESRVAYYC_optimization2_prediction.csv')


from copy import deepcopy
seq_for_optimization = ['CTESQGSLC']
optional_seqs = []
for seq in seq_for_optimization:
    for i in range(1, 8):
        for symbol in BASE_AMINO_ACIDS:
            list_of_seq = deepcopy(list(seq))
            list_of_seq[i] = symbol
            list_of_seq = ''.join(list_of_seq)
            if list_of_seq not in optional_seqs:
                optional_seqs.append(list_of_seq)

model = model.eval()
option_seqs_embeddings = compute_esm_embedding(optional_seqs, 'sequence')
prediction = model(option_seqs_embeddings)
output_df = pd.DataFrame({'seq': optional_seqs,
                          'prediction': prediction.detach().squeeze().numpy().tolist()})
output_df.to_csv('/mnt/c/tmp/CTESQGSLC_optimization2_prediction.csv')


from copy import deepcopy
seq_for_optimization = ['CQEGVLARC']
optional_seqs = []
for seq in seq_for_optimization:
    for i in range(1, 8):
        for symbol in BASE_AMINO_ACIDS:
            list_of_seq = deepcopy(list(seq))
            list_of_seq[i] = symbol
            list_of_seq = ''.join(list_of_seq)
            if list_of_seq not in optional_seqs:
                optional_seqs.append(list_of_seq)

model = model.eval()
option_seqs_embeddings = compute_esm_embedding(optional_seqs, 'sequence')
prediction = model(option_seqs_embeddings)
output_df = pd.DataFrame({'seq': optional_seqs,
                          'prediction': prediction.detach().squeeze().numpy().tolist()})
output_df.to_csv('/mnt/c/tmp/CQEGVLARC_optimization2_prediction.csv')


from copy import deepcopy
seq_for_optimization = ['CESRVAYYC']
optional_seqs = ['CESRVAYYC']
for seq in seq_for_optimization:
    for i in range(1, 8):
        for symbol in ['A']:
            list_of_seq = deepcopy(list(seq))
            list_of_seq[i] = symbol
            list_of_seq = ''.join(list_of_seq)
            if list_of_seq not in optional_seqs:
                optional_seqs.append(list_of_seq)

model = model.eval()
option_seqs_embeddings = compute_esm_embedding(optional_seqs, 'sequence')
prediction = model(option_seqs_embeddings)
output_df = pd.DataFrame({'seq': optional_seqs,
                          'prediction': prediction.detach().squeeze().numpy().tolist()})
output_df.to_csv('/mnt/c/tmp/CESRVAYYC_optimization_prediction.csv')

from copy import deepcopy
seq_for_optimization = ['CTESQGSLC']
optional_seqs = ['CTESQGSLC']
for seq in seq_for_optimization:
    for i in range(1, 8):
        for symbol in ['A']:
            list_of_seq = deepcopy(list(seq))
            list_of_seq[i] = symbol
            list_of_seq = ''.join(list_of_seq)
            if list_of_seq not in optional_seqs:
                optional_seqs.append(list_of_seq)

model = model.eval()
option_seqs_embeddings = compute_esm_embedding(optional_seqs, 'sequence')
prediction = model(option_seqs_embeddings)
output_df = pd.DataFrame({'seq': optional_seqs,
                          'prediction': prediction.detach().squeeze().numpy().tolist()})
output_df.to_csv('/mnt/c/tmp/CTESQGSLC_optimization_prediction.csv')

from copy import deepcopy
seq_for_optimization = ['CQEGVLARC']
optional_seqs = ['CQEGVLARC']
for seq in seq_for_optimization:
    for i in range(1, 8):
        for symbol in ['A']:
            list_of_seq = deepcopy(list(seq))
            list_of_seq[i] = symbol
            list_of_seq = ''.join(list_of_seq)
            if list_of_seq not in optional_seqs:
                optional_seqs.append(list_of_seq)

model = model.eval()
option_seqs_embeddings = compute_esm_embedding(optional_seqs, 'sequence')
prediction = model(option_seqs_embeddings)
output_df = pd.DataFrame({'seq': optional_seqs,
                          'prediction': prediction.detach().squeeze().numpy().tolist()})
output_df.to_csv('/mnt/c/tmp/CQEGVLARC_optimization_prediction.csv')