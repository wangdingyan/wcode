import requests as r
import sys

sys.path.append('/home/fluoxetine/transformerCPI2.0')
from featurizer import featurizer

import torch
import random
import os
import numpy as np
########################################################################################################################

class Tester(object):
    def __init__(self, model,device):
        self.model = model
        self.device = device
    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            for data in dataset:
                adjs, atoms, proteins = [], [], []
                atom, adj, protein= data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                data = pack(atoms,adjs,proteins, self.device)
                predicted_scores = self.model(data)
        return predicted_scores

def pack(atoms, adjs, proteins, device):
    atoms = torch.FloatTensor(atoms)
    adjs = torch.FloatTensor(adjs)
    proteins = torch.FloatTensor(proteins)
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0]+1)
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]
    atoms_len += 1
    protein_num = []
    for protein in proteins:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
    atoms_new = torch.zeros((N,atoms_len,34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, 1:a_len+1, :] = atom
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        adjs_new[i,0,:] = 1
        adjs_new[i,:,0] = 1
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        adjs_new[i, 1:a_len+1, 1:a_len+1] = adj
        i += 1
    proteins_new = torch.zeros((N, proteins_len),dtype=torch.int64, device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len] = protein
        i += 1
    return (atoms_new, adjs_new, proteins_new, atom_num, protein_num)



def seq2file(uniprot_id):
    baseUrl="http://www.uniprot.org/uniprot/"
    currentUrl=baseUrl+uniprot_id+".fasta"
    response = r.post(currentUrl)
    cData=''.join(response.text)
    cData=''.join(cData.split('\n')[1:])
    return cData

def transformercpi(uniprot_id, smiles):
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    model = torch.load('/home/fluoxetine/transformerCPI2.0/Virtual Screening.pt')  # Load trained model
    model.to(device)
    sequence = seq2file(uniprot_id)
    compounds, adjacencies, proteins = featurizer(smiles, sequence)
    tester = Tester(model, device)
    test_set = list(zip(compounds, adjacencies, proteins))
    score = float(tester.test(test_set))
    return score


if __name__ == "__main__":
    # import pandas as pd
    # df_smiles = pd.read_excel('/mnt/c/tmp/20240629_1.xlsx', index_col='index')
    # df_targets = pd.read_excel('/mnt/c/tmp/20240629_2.xlsx', index_col=None)
    # smiles_score = []
    # full_smiles_score = []
    # for t in df_targets.iterrows():
    #     index_num = t[1]['index']
    #     uniprot_id = t[1]['uniprot_id']
    #     s = transformercpi(uniprot_id, df_smiles.iloc[index_num-1]['smiles'])
    #     s_full = transformercpi(uniprot_id, df_smiles.iloc[index_num-1]['full_smiles'])
    #     smiles_score.append(s)
    #     full_smiles_score.append(s_full)
    #     print(index_num, uniprot_id, s, s_full)
    #
    # df_targets['score'] = smiles_score
    # df_targets['full_score'] = full_smiles_score
    # df_targets.to_excel('/mnt/c/tmp/20240629_3.xlsx')
    #

    import pandas as pd
    df = pd.read_excel('/mnt/c/tmp/xmjsjhz.xlsx')
    targets = df['Targets'].tolist()
    scores = []
    for t in targets:
        s = transformercpi(t, 'COC1=C(C2=C[N+]3=C(C=C2C=C1)C4=CC5=C(C=C4CC3)OCO5)OC')
        print(t, s)
        scores.append(s)
    df['scores'] = scores
    df.to_excel('/mnt/c/tmp/xmjsjhz2.xlsx')
    # print(transformercpi('P05108', 'C[C@@H]1CC[C@H]2[C@H](C(=O)O[C@H]3[C@@]24[C@H]1CC[C@](O3)(OO4)C)C'))
