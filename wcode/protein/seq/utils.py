"""Utility functions for sequence-based featurisation."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import Any, Callable, List

import networkx as nx
import numpy as np
########################################################################################################################
def parse_fasta(file_path):
    sequences = []
    current_sequence = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # This is a header line
                if current_sequence:
                    sequences.append(current_sequence)
                current_sequence = {'header': line[1:], 'sequence': ''}
            else:
                # This is a sequence line
                if current_sequence is not None:
                    current_sequence['sequence'] += line

        # Don't forget to add the last sequence after loop ends
        if current_sequence:
            sequences.append(current_sequence)

    # Extract sequences from the dictionary structure
    for seq in sequences:
        seq['sequence'] = seq['sequence'].upper()  # Convert sequence to uppercase if needed

    return sequences

########################################################################################################################
def compute_feature_over_chains(
    G: nx.Graph, func: Callable, feature_name: str
) -> nx.Graph:
    """
    Computes a sequence featurisation function over the chains in a graph.

    :param G: nx.Graph protein structure graph to featurise.
    :type G: nx.Graph
    :param func: Sequence featurisation function.
    :type func: Callable
    :param feature_name: name of added feature.
    :type feature_name: str
    :return: Graph with added features of the form ``G.graph[f"{feature_name}_{chain_id}"]``.
    :rtype: nx.Graph
    """
    for c in G.graph["chain_ids"]:
        G.graph[f"{feature_name}_{c}"] = func(G.graph[f"sequence_{c}"])
        """
        feat = func(G.graph[f"sequence_{c}"])

        if out_type == "series":
            feat = pd.Series(feat)
        elif out_type == "np":
            raise NotImplementedError

        G.graph[f"{feature_name}_{c}"] = feat
        """
    return G


def aggregate_feature_over_chains(
    G: nx.Graph, feature_name: str, aggregation_type: str
) -> nx.Graph:
    """
    Performs aggregation of a given feature over chains in a graph to produce an aggregated value.

    :param G: nx.Graph protein structure graph.
    :type G: nx.Graph
    :param feature_name: Name of feature to aggregate.
    :type feature_name: str
    :param aggregation_type: Type of aggregation to perform (``"min"``, ``"max"``, ``"mean"``, ``"sum"``).
    :type aggregation_type: str
    :raises ValueError: If ``aggregation_type`` is not one of ``"min"``, ``"max"``, ``"mean"``, ``"sum"``.
    :return: Graph with new aggregated feature.
    :rtype: nx.Graph
    """
    func = parse_aggregation_type(aggregation_type)

    G.graph[f"{feature_name}_{aggregation_type}"] = func(
        [G.graph[f"{feature_name}_{c}"] for c in G.graph["chain_ids"]], axis=0
    )
    return G


def aggregate_feature_over_residues(
    G: nx.Graph,
    feature_name: str,
    aggregation_type: str,
) -> nx.Graph:
    """
    Performs aggregation of a given feature over chains in a graph to produce an aggregated value.

    :param G: nx.Graph protein structure graph.
    :type G: nx.Graph
    :param feature_name: Name of feature to aggregate.
    :type feature_name: str
    :param aggregation_type: Type of aggregation to perform (``"min"``, ``"max"``, ``"mean"``, ``"sum"``).
    :type aggregation_type: str
    :raises ValueError: If ``aggregation_type`` is not one of ``"min"``, ``"max"``, ``"mean"``, ``"sum"``.
    :return: Graph with new aggregated feature.
    :rtype: nx.Graph
    """
    func = parse_aggregation_type(aggregation_type)

    for c in G.graph["chain_ids"]:
        chain_features = []
        for n in G.nodes:
            if G.nodes[n]["chain_id"] == c:
                chain_features.append(G.nodes[n][feature_name])
        G.graph[f"{feature_name}_{aggregation_type}_{c}"] = func(
            chain_features, axis=0
        )
    return G


def sequence_to_ngram(sequence: str, N: int) -> List[str]:
    """
    Chops a sequence into overlapping N-grams (substrings of length ``N``).

    :param sequence: str Sequence to convert to N-grams.
    :type sequence: str
    :param N: Length of N-grams.
    :type N: int
    :return: List of n-grams.
    :rtype: List[str]
    """
    return [sequence[i : i + N] for i in range(len(sequence) - N + 1)]


def subset_by_node_feature_value(
    G: nx.Graph, feature_name: str, feature_value: Any
):
    """
    Extracts a subgraph from a protein structure graph based on nodes with a certain feature value.

    :param G: nx.Graph protein structure graph to extract a subgraph from.
    :type G: nx.Graph
    :param feature_name: Name of feature to base subgraph extraction from.
    :type feature_name: str
    :param feature_value: Value of feature to select.
    :type feature_value: Any
    :return: Subgraph of ``G`` based on nodes with a given feature value.
    :rtype: nx.Graph
    """
    node_list = [
        n for n, d in G.nodes(data=True) if d[feature_name] == feature_value
    ]
    residue_list = []
    for n in node_list:
        residue_id = G.nodes[n]['residue_id']
        if residue_id not in residue_list:
            residue_list.append(residue_id)

    g = G.subgraph(node_list)
    return g, residue_list

def parse_aggregation_type(aggregation_type) -> Callable:
    """Returns an aggregation function by name

    :param aggregation_type: One of: ``["max", "min", "mean", "median", "sum"]``.
    :type aggregration_type: AggregationType
    :returns: NumPy aggregation function.
    :rtype: Callable
    :raises ValueError: if aggregation type is not supported.
    """
    if aggregation_type == "max":
        func = np.max
    elif aggregation_type == "min":
        func = np.min
    elif aggregation_type == "mean":
        func = np.mean
    elif aggregation_type == "median":
        func = np.median
    elif aggregation_type == "sum":
        func = np.sum
    else:
        raise ValueError(
            f"Unsupported aggregator: {aggregation_type}."
            f" Please use min, max, mean, median, sum"
        )
    return func