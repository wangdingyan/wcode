import numpy as np
from torch.nn import Sigmoid
import os
import torch
from glob import glob
from wcode.model.GINEConv import GraphEmbeddingModel
from torch_geometric.data import Batch
from torch.nn import Linear, Module, BCELoss
from torch.optim import Adam
from tqdm import tqdm

class RRedockDataset():
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.case_names = self.get_case_names_with_positive_sample_success()

    def get_case_names(self):
        return os.listdir(self.base_dir)

    def get_sample_names(self, case_name):
        return os.listdir(os.path.join(self.base_dir, case_name))

    def get_sample_files_paths(self, case_name, sample_name):
        return glob(os.path.join(self.base_dir, case_name, sample_name, '*'))

    def get_all_sample_files_paths(self):
        all_file_paths = []
        for cn in self.get_case_names():
            for sample in self.get_sample_names(cn):
                fs = self.get_sample_files_paths(cn, sample)
                fs = [f for f in fs if 'merge' not in f]
                fs = [f for f in fs if not f.endswith('.pt')]
                f1, f2 = fs
                if f1.endswith('pdb'):
                    pdb_file, ligand_file = f1, f2
                else:
                    pdb_file, ligand_file = f2, f1
                all_file_paths.append((pdb_file, ligand_file))
        return all_file_paths

    def feature_extraction_progress(self):
        case_name_list = self.get_case_names()
        for c_n in case_name_list:
            sample_paths = self.get_sample_names(c_n)
            finished_feature_extraction = glob(os.path.join(self.base_dir, c_n, "*", "*.pt"))
            num_pos = np.sum([f.split('\\')[-2] == f.split('\\')[-3] for f in finished_feature_extraction])
            print(f'{c_n}\t{len(sample_paths)}\t{len(finished_feature_extraction)}\t{num_pos}')

    def get_case_names_with_positive_sample_success(self):
        case_name_list = self.get_case_names()
        output_list = []
        for c_n in case_name_list:
            finished_feature_extraction = glob(os.path.join(self.base_dir, c_n, "*", "*.pt"))
            num_pos = np.sum([f.split('\\')[-2] == f.split('\\')[-3] for f in finished_feature_extraction])
            if num_pos > 0:
                output_list.append(c_n)
        return output_list

    @staticmethod
    def metric_top_rank(predictions, labels):
        assert np.sum(labels) == 1
        rank = np.argmax(labels)
        p_rank = np.argsort(predictions)[::-1]
        return p_rank.tolist().index(rank)

    def get_case_feature_and_label(self, case_name):
        features = glob(os.path.join(self.base_dir, case_name, "*", "*.pt"))
        labels = np.array([f.split('\\')[-2] == f.split('\\')[-3] for f in features]).astype(np.int64)
        return features, labels

    def __getitem__(self, item):
        return self.get_case_feature_and_label(self.case_names[item])


class scoring_model(Module):
    def __init__(self):
        super().__init__()
        self.embedding = GraphEmbeddingModel(142,
                                             5,
                                             0,
                                             128,
                                             100)
        self.output_layer = Linear(100, 1)
        self.activation = Sigmoid()

    def forward(self, feature_paths_list):
        g = Batch.from_data_list([torch.load(n) for n in feature_paths_list])
        atom_feature = torch.concat([g.residue_name_one_hot,
                                     g.atom_type_one_hot,
                                     g.record_symbol_one_hot,
                                     g.rdkit_atom_feature_onehot], dim=-1).float().cuda()
        edge_index = torch.Tensor(g.edge_index).long().cuda()
        bond_feature = torch.Tensor(g.bond_feature).float().cuda()
        embedding = self.embedding(atom_feature,
                                   edge_index,
                                   bond_feature,
                                   node2graph=g.batch.cuda())[1]
        output = self.activation(self.output_layer(embedding).squeeze())
        return output


rredockdataset = RRedockDataset('D:\\pdb_sdf_dataset_test')
dataset_cases =rredockdataset.get_case_names_with_positive_sample_success()

train_cases = dataset_cases[:int(0.8 * len(dataset_cases))]
test_cases =dataset_cases[int(0.8 * len(dataset_cases)):]
print(f'Length of Training Set: {len(train_cases)}')
print(f'Length of Test Set:     {len(test_cases)}')

model = scoring_model()
model = model.cuda()
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = BCELoss()

for epoch in range(100):
    with torch.no_grad():
        torch.save(model.cpu().state_dict(), "C:\\tmp\\pocket_prediction_model.pt")
        model.cuda()
        model.eval()
        rank = []
        for case_name in tqdm(test_cases):
            f, l = rredockdataset.get_case_feature_and_label(case_name)
            p = model(f).unsqueeze(0)
            l = torch.from_numpy(l).cuda().unsqueeze(0).float()
            l = l.detach().cpu().squeeze().tolist()
            p = p.detach().cpu().squeeze().tolist()
            rank.append(RRedockDataset.metric_top_rank(p, l))
        print(np.mean(rank))

    model.train()
    for case_name in tqdm(train_cases):
        f, l = rredockdataset.get_case_feature_and_label(case_name)
        p = model(f).unsqueeze(0)
        l = torch.from_numpy(l).cuda().unsqueeze(0).float()
        loss = loss_fn(p, l)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()






