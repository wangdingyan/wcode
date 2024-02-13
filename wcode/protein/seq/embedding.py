# https://github.com/a-r-j/graphein/blob/master/graphein/protein/features/sequence/embeddings.py

import os
from functools import lru_cache, partial
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import esm
from wcode.protein.seq.utils import subset_by_node_feature_value


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
    sequence: str,
    representation: str,
    output_layer: int = 33,
) -> np.ndarray:
    model, alphabet = _load_esm_model()
    batch_converter = alphabet.get_batch_converter()

    data = [
        ("protein1", sequence),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(
            batch_tokens, repr_layers=[output_layer], return_contacts=True
        )
    token_representations = results["representations"][output_layer]

    if representation == "residue":
        return token_representations.numpy()

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first
    # residue is token 1.
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )
        return sequence_representations[0].numpy()


def _load_esm_model():
    return esm.pretrained.esm1b_t33_650M_UR50S()

if __name__ == '__main__':
    print(compute_esm_embedding('AAAA', 'sequence'))