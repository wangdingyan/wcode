# https://github.com/a-r-j/graphein/blob/master/graphein/protein/features/sequence/embeddings.py

from typing import List
import networkx as nx
import numpy as np
import torch
import esm
from tqdm import tqdm

try:
    from esme import ESM2
    from esme.alphabet import tokenize
except:
    pass
from wcode.protein.constant import STANDARD_AMINO_ACID_MAPPING_3_TO_1
from wcode.protein.seq.utils import subset_by_node_feature_value
from copy import deepcopy


def esm_residue_embedding(
    G: nx.Graph,
    output_layer: int = 33
) -> nx.Graph:
    for chain in G.graph["chain_ids"]:
        embedding = compute_esm_embedding(
            G.graph[f"sequence_{chain}"],
            representation="residue",
            output_layer=output_layer,
        )
        # remove start and end tokens from per-token residue embeddings
        embedding = embedding[0, 1:-1]
        subgraph, residue_list = subset_by_node_feature_value(G, "chain_id", chain)

        for i, (n, d) in enumerate(subgraph.nodes(data=True)):
            if d['record_name'] != "ATOM":
                continue
            residue_id = G.nodes[n]['residue_id']
            idx = residue_list.index(residue_id)
            G.nodes[n]["esm_embedding"] = embedding[idx]

    return G


def compute_esm_embedding(
    sequences: List[str],
    representation: str,
    output_layer: int = 33,
    contacts: bool = False,
) -> np.ndarray:
    model, alphabet = _load_esm_model()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # data = [
    #     ("protein1", sequence),
    # ]
    data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(
            batch_tokens, repr_layers=[output_layer], return_contacts=True
        )

    token_representations = results["representations"][output_layer]
    contact_array = results["contacts"]

    if representation == "residue":
        if not contacts:
            return token_representations.numpy()
        else:
            return token_representations.numpy(), contact_array

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first
    # residue is token 1.
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1: len(seq) + 1].mean(0)
            )
        if not contacts:
            return torch.stack(sequence_representations)
        else:
            return torch.stack(sequence_representations), contact_array


def _load_esm_model():
    return esm.pretrained.esm2_t33_650M_UR50D()


def esme_embedding(sequences: List[str],
                   batch_size=500,
                   representation='sequence',
                   device=0) -> torch.Tensor:


    model = ESM2.from_pretrained("/home/wang_ding_yan/model/esm2_3b.safetensors", device=device)
    total_embedding = []
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_sequences = sequences[i:i + batch_size]
        tokens = tokenize(batch_sequences).to(device)
        sequence_embeddings = model.embedding(tokens)[:, 1:-1, :]
        total_embedding.append(sequence_embeddings)
    total_embedding = torch.cat(total_embedding, dim=0)
    if representation == "sequence":
        return total_embedding.mean(1).float().detach()
    else:
        return total_embedding.float().detach()


if __name__ == '__main__':
    print(esme_embedding(['CCCCCC'], representation='sequence').shape)
    # (1, 8, 1280)
    print(esme_embedding(['AAAA', 'CCCCCC'], representation='sequence').shape)
    # (2, 8, 1280)
    # print(esme_embedding(['DASP GLU HIS', 'ALA DGLU HIS PRO']*200, device=3).shape)