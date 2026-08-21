import numpy as np
from Bio import Phylo

def create_maps(all_nodes, tree):
    all_branches = []
    branch_lengths = {}
    for n, node in enumerate(all_nodes):
        if node.name != tree.root.name:
            all_branches.append(node.name)
            branch_lengths[node.name] = node.branch_length
    return all_branches, branch_lengths

def get_path(spA, spB, tree):
    MRCA = tree.common_ancestor(spA,spB)
    negative_path = [node.name for node in MRCA.get_path(spA)[::-1]]
    positive_path = [node.name for node in MRCA.get_path(spB)]
    path = negative_path + positive_path
    return path

def fetch_data(spA, spB, data):
    if (spA,spB) in data.keys():
        return data[(spA,spB)]
    elif (spB,spA) in data.keys():
        return data[(spB,spA)]
    else:
        raise ValueError("Distribution for %s-%s comparison is not here..."%(spA,spB))

def convert_two_parameters_to_full(gamma_theta, n_branches):
    gamma, theta = gamma_theta
    x = np.zeros(2 * int(n_branches))
    x[::2] = gamma
    x[1::2] = theta
    return x
