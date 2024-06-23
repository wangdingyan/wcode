import pandas as pd
import numpy as np
from torch_geometric.data import Batch, Data, Dataset  # , InMemoryDataset
from torch.nn.utils.rnn import pad_sequence
import torch
from collections import defaultdict
from copy import deepcopy
import copy

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

def sequence_to_one_hot(sequence, aa_to_index):
    indices = [aa_to_index[aa] for aa in sequence]
    one_hot = torch.zeros(len(sequence), len(amino_acids))
    one_hot.scatter_(1, torch.tensor(indices).unsqueeze(1), 1)
    return one_hot

class CycPepDataset(Dataset):

    def __init__(self, data=None, transform=None):
        super().__init__()
        self.data = data
        self.names = sorted(list(self.data.keys()))
        self.transform = transform

    def __getitem__(self, idx):

        data = deepcopy(self.data[self.names[idx]])
        # print(data)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        seqs  = [s['seq'] for s in batch]
        bonds = [torch.Tensor(s['bond']) for s in batch]
        angles = [torch.Tensor(s['angle']) for s in batch]
        dihedrals = [torch.Tensor(s['dihedral']) for s in batch]

        seqs = [sequence_to_one_hot(seq, aa_to_index) for seq in seqs]
        pad_seqs = pad_sequence(seqs).permute([1,0,2])
        pad_bonds = pad_sequence(bonds).permute([1,0])
        pad_angles = pad_sequence(angles).permute([1,0])
        pad_dihedrals = pad_sequence(dihedrals).permute([1,0])

        return pad_seqs, pad_bonds, pad_angles, pad_dihedrals

class CycPepDataset_PackedConf(CycPepDataset):

    def __init__(self, data=None, transform=None):
        super(CycPepDataset_PackedConf, self).__init__(data, transform)
        self._pack_data_by_mol()

    def _pack_data_by_mol(self):
        """
        pack confs with same mol into a single data object
        """
        self._packed_data = defaultdict(list)
        if hasattr(self.data, 'idx'):
            for i in range(len(self.data)):
                self._packed_data[self.data[i].idx.item()].append(self.data[i])
        else:
            for i in range(len(self.data)):
                self._packed_data[self.data[i].smiles].append(self.data[i])
        print('got %d molecules with %d confs' % (len(self._packed_data), len(self.data)))

        new_data = []
        # logic
        # save graph structure for each mol once, but store all confs
        cnt = 0
        for k, v in self._packed_data.items():
            data = copy.deepcopy(v[0])
            all_pos = []
            for i in range(len(v)):
                all_pos.append(v[i].pos)
            data.pos_ref = torch.cat(all_pos, 0)  # (num_conf*num_node, 3)
            data.num_pos_ref = torch.tensor([len(all_pos)], dtype=torch.long)
            # del data.pos

            if hasattr(data, 'totalenergy'):
                del data.totalenergy
            if hasattr(data, 'boltzmannweight'):
                del data.boltzmannweight
            new_data.append(data)
        self.new_data = new_data

    def __getitem__(self, idx):

        data = self.new_data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.new_data)