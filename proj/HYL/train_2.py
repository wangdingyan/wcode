import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from wcode.protein.seq.embedding import esme_embedding, compute_esm_embedding
from wcode.cycpep.data import generate_cyclic_permutations
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
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


def collate_fn(batch):
    seqs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    seq_embeddings = [item[2] for item in batch]

    return torch.stack(seq_embeddings), torch.Tensor(labels)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(2560, 512)
        self.linear2 = torch.nn.Linear(512, 128)
        self.linear3 = torch.nn.Linear(128, 1)
        self.activation = torch.nn.ReLU()
        self.output = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.output(x)
        return x.squeeze()


# class Model(nn.Module):
#     def __init__(self,
#                  embedding_size=2560,
#                  num_heads=8,
#                  num_layers=2,
#                  hidden_size=512,
#                  num_classes=2,
#                  dropout=0.3):
#         super(Model, self).__init__()
#
#         # 参数
#         self.embedding_size = embedding_size
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.num_classes = num_classes
#
#         # Dropout层
#         self.dropout = nn.Dropout(dropout)
#
#         # 定义一个TransformerEncoder
#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embedding_size,
#             nhead=num_heads,
#             dim_feedforward=hidden_size,
#             dropout=dropout
#         )
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#
#         # 分类层：全连接层将Transformer输出映射到类别数
#         self.fc = nn.Linear(embedding_size, num_classes)
#
#         # LayerNorm：用于全连接层输出之前的正则化
#         self.layer_norm = nn.LayerNorm(embedding_size)
#
#     def forward(self, x):
#         # x的形状是 [batch_size, sequence_length, embedding_size]
#
#         # 将输入调整为 [sequence_length, batch_size, embedding_size]
#         x = x.transpose(0, 1)
#
#         # Transformer的输入是[sequence_length, batch_size, embedding_size]
#         transformer_out = self.transformer_encoder(x)
#
#         # 获取最后一个时间步的输出，作为整个句子的表示
#         # 输出形状 [batch_size, embedding_size]
#         sentence_rep = transformer_out[-1, :, :]
#
#         # 正则化：LayerNorm
#         sentence_rep = self.layer_norm(sentence_rep)
#
#         # Dropout层
#         sentence_rep = self.dropout(sentence_rep)
#
#         # 通过全连接层得到分类输出
#         output = self.fc(sentence_rep)
#
#         # 输出一个二分类的概率值
#         return torch.sigmoid(output[:, 1])
def batch_iter(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


if __name__ == '__main__':
    from copy import deepcopy
    best_test_roc_list = []
    for round, random_seed in enumerate([1111,2222,3333,4444,5555,6666,7777,8888,9999]):
        train_df = pd.read_csv('/home/wang_ding_yan/data/cycpep/cyc7_2.csv')
        train_items = list(train_df.iterrows())
        train_sequences = [s[1]['seq'] for s in train_items]
        train_label = [s[1]['label'] for s in train_items]

        train_sequences, test_sequences, train_label, test_label = train_test_split(train_sequences, train_label, test_size=0.2, random_state=random_seed)
        # for s, l in zip(deepcopy(train_sequences), (train_label)):
        #     additional_seq = generate_cyclic_permutations(s, 5)
        #     for a_s in additional_seq:
        #         train_label.append(l)

        print(len(train_sequences), len(test_sequences), train_sequences[0], test_sequences[0])
        train_dataset = [[s, int(l > 0.96)] for s, l in zip(train_sequences, train_label)]
        test_dataset = [[s, int(l > 0.96)] for s, l in zip(test_sequences, test_label)]
        print(np.sum(np.array(train_label) > 0.96), np.sum(np.array(test_label) > 0.96))

        # training_esm_embeddings, training_cm_embeddings = compute_esm_embedding(train_sequences, representation='sequence')
        # test_esm_embeddings, test_cm_embeddings = compute_esm_embedding(test_sequences, representation='sequence')
        # training_cm_embeddings = training_cm_embeddings.reshape([training_cm_embeddings.shape[0], -1])
        # test_cm_embeddings = test_cm_embeddings.reshape([test_cm_embeddings.shape[0], -1])
        #
        # training_embeddings = torch.cat([training_esm_embeddings, training_cm_embeddings], dim=1)
        # test_embeddings = torch.cat([test_esm_embeddings, test_cm_embeddings], dim=1)

        training_esm_embeddings = esme_embedding(train_sequences, representation='sequence',device=3)
        test_esm_embeddings = esme_embedding(test_sequences, representation='sequence',device=3)
        print(training_esm_embeddings.shape, test_esm_embeddings.shape)

        training_embeddings = training_esm_embeddings
        test_embeddings = test_esm_embeddings
        print(training_embeddings.shape, test_embeddings.shape)

        for item, emb in zip(train_dataset, training_embeddings):
            item.append(emb)

        for item, emb in zip(test_dataset, test_embeddings):
            item.append(emb)

        training_dataset = CustomDataset(train_dataset)
        test_dataset = CustomDataset(test_dataset)

        model = Model().to(3)
        loss_fn = nn.BCELoss()
        # loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        train_loader = torch.utils.data.DataLoader(training_dataset, collate_fn=collate_fn, batch_size=48, shuffle=True,
                                                   drop_last=True)
        train_noshuffle_loader = torch.utils.data.DataLoader(training_dataset, collate_fn=collate_fn, batch_size=48,
                                                             shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=48, shuffle=False)

        best_roc = 0
        best_roc_not_change = 0
        for epoch in range(1000):
            model = model.eval()

            # test set evaluation
            test_predictions = []
            test_labels = []

            for batch in test_loader:
                predictions = model(batch[0])
                labels = batch[1]
                test_predictions.extend(predictions.cpu().detach().numpy().tolist())
                test_labels.extend(labels.detach().numpy().tolist())

            test_loss = loss_fn(torch.Tensor(test_predictions), torch.Tensor(test_labels))
            test_metrics = roc_auc_score(np.array(test_labels), np.array(test_predictions))
            # test_metrics = spearmanr(np.array(test_labels), np.array(test_predictions))[0]
            if test_metrics > best_roc:
                best_roc = test_metrics
                best_roc_not_change = 0
            else:
                best_roc_not_change += 1

            print(f"Test Loss {test_loss:.3f}, Test Metrics {test_metrics:.3f}, Best Metrics {best_roc:.3f}, Best_roc_not_change {best_roc_not_change:.3f}")

            df = pd.DataFrame({'test_sequences': test_sequences,
                               'test_labels': test_labels,
                               'test_predictions': test_predictions})
            df.to_csv('/home/wang_ding_yan/tmp/test_set_prediction.csv')
            if best_roc_not_change >= 100:
                best_test_roc_list.append(best_roc)
                break


            train_predictions = []
            train_labels = []
            for batch in train_noshuffle_loader:
                predictions = model(batch[0])
                labels = batch[1]
                train_predictions.extend(predictions.cpu().detach().numpy().tolist())
                train_labels.extend(labels.detach().numpy().tolist())

            train_loss = loss_fn(torch.Tensor(train_predictions), torch.Tensor(train_labels))
            train_metrics = roc_auc_score(np.array(train_labels), np.array(train_predictions))
            # train_metrics = spearmanr(np.array(train_labels), np.array(train_predictions))[0]
            print(f"Train Loss {train_loss:.3f}, Train Metrics {train_metrics:.3f}")

            df = pd.DataFrame({'train_sequences': train_sequences,
                               'train_labels': train_labels,
                               'train_predictions': train_predictions})
            df.to_csv('/home/wang_ding_yan/tmp/train_set_prediction.csv')

            model = model.train()
            model.zero_grad()

            for batch in train_loader:
                predictions = model(batch[0]).to(3)
                labels = batch[1].to(3)
                weight = []
                for l in labels:
                    if l == 1:
                        weight.append(20)
                    else:
                        weight.append(1)
                weight = torch.Tensor(weight).to(3)
                loss_fn2 = nn.BCELoss(weight=weight)
                loss = loss_fn2(predictions, labels)
                # loss_fn1 = nn.BCELoss().to(3)
                loss = loss_fn2(predictions, labels)
                loss.backward()
                optimizer.step()
                model.zero_grad()

        torch.save(model.state_dict(), '/home/wang_ding_yan/tmp/model_save.pt')
        model = model.eval()
        print(f"ROUND {round}: Mean auROC {np.mean(best_test_roc_list):.3f}+-{np.std(best_test_roc_list):.3f}")


