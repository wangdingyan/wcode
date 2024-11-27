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
) -> np.ndarray:
    model, alphabet = _load_esm_model()
    batch_converter = alphabet.get_batch_converter()

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

    if representation == "residue":
        return token_representations.numpy()

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first
    # residue is token 1.
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1: len(seq) + 1].mean(0)
            )
        return torch.stack(sequence_representations)


def _load_esm_model():
    return esm.pretrained.esm2_t33_650M_UR50D()


if __name__ == '__main__':
    print(compute_esm_embedding(['CCCCCC'], 'residue').shape)
    # (1, 8, 1280)
    print(compute_esm_embedding(['AAAA', 'CCCCCC'], 'residue').shape)
    # (2, 8, 1280)


def compute_sequence_embeddings_fast(sequences: List[str],
                                     batch_size=1000) -> torch.Tensor:
    with torch.no_grad():
        model = ESM2.from_pretrained("/mnt/d/tmp/650M.safetensors", device=0)
        max_length = max(len(s.split()) for s in sequences)
        total_embedding = torch.zeros(len(sequences), max_length, 35, dtype=torch.float32)

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
            tokens = tokenize(batch_sequences)
            sequence_embeddings = model(tokens.cuda())[:, 1:-1, :]

            for j, embedding in enumerate(sequence_embeddings):
                total_embedding[i + j, :embedding.size(0), :33] = embedding

        return total_embedding
