import numpy as np
from scipy import linalg, stats, optimize
from Bio import Phylo
import networkx as ntx
from io import StringIO

def random_tree(N, L):
    tree_string = "(*1*)"
    nodes = ["1"]
    n = 1
    while n < N:
        old_node = np.random.choice(nodes)
        max_node = np.max([int(i) for i in nodes])
        new_nodeA = str(max_node + 1)
        new_nodeB = str(max_node + 2)
        tree_string = tree_string.replace('*'+old_node+'*',"(*%s*,*%s*)"%(new_nodeA,new_nodeB))
        nodes = tree_string.replace('*','').replace('(','').replace(')','').rstrip(';').split(',')        
        n = len(nodes)
    tree_string = tree_string.replace('*','') + ';'
    tree = Phylo.read(StringIO(tree_string),'newick')
    all_nodes0 = tree.get_terminals() + tree.get_nonterminals()
    for node in all_nodes0:
        if node != tree.root:
            node.branch_length = 1
    tree.collapse(tree.root[0])
     
    all_nodes = tree.get_terminals() + tree.get_nonterminals()
    for name, node in enumerate(all_nodes):
        node.name = str(name + 1)
        if node != tree.root:
            r = np.random.rand()
            d = len(tree.root.get_path(node))
            node.branch_length = L * (r / d)
    return tree

def generate_rates(tree, max_rate):
    all_nodes = tree.get_terminals() + tree.get_nonterminals()
    rates = {}
    for node in all_nodes:
        if node != tree.root:
            rates[node.name] = np.random.rand() * max_rate
    return rates

def generate_initial_genome(chromosomes, genes_per_chromosome):
    karyotype = chromosomes * [genes_per_chromosome]
    n_genes = np.sum(karyotype)
    chrom_size = karyotype[0]
    nodes = np.arange(1,n_genes+1).astype(int)
    edges = []
    edge_counter = 0
    for node in nodes:
        if (node % chrom_size != 0):
            edges.append([node,node+1])
            edge_counter += 1
    edges = np.vstack(edges)
    edge_list = [(edge[0],edge[1]) for edge in edges]
    
    G0 = ntx.Graph(edge_list)
    return G0

def generate_edge_weights_discrete(G, p_HC, p_CH, w_C, w_H):
    weights = {}
    for chrom in ntx.connected_components(G):
        Gc = ntx.subgraph(G,chrom)
        D = dict(ntx.degree(Gc))
        ends = [node for node in D.keys() if D[node] == 1]
        path = list(ntx.edge_dfs(Gc,source=ends[0]))
        w = [np.random.choice([w_C, w_H])]
        weights[path[0]] = w[-1]
        for edge in path[1:]:
            r = np.random.rand()
            if w[-1] == w_C:
                if r < p_CH:
                    weights[edge] = w_H
                else:
                    weights[edge] = w_C
            elif w[-1] == w_H:
                if r < p_HC:
                    weights[edge] = w_C
                else:
                    weights[edge] = w_H               
            w.append(weights[edge])
    return weights

def find_constrained_blocks(weights, nchromosomes, ngenes_per_chromosome, max_weight = 1):
    cold_blocks = []
    weights_array = np.array([weights[edge] for edge in weights.keys()])
    edge_list = list([edge for edge in weights.keys()])
    for nchrom in range(nchromosomes):
        chrom_weights = weights_array[nchrom*(ngenes_per_chromosome-1):(nchrom+1)*(ngenes_per_chromosome-1)]
        chrom_edges = edge_list[nchrom*(ngenes_per_chromosome-1):(nchrom+1)*(ngenes_per_chromosome-1)]
        where_cold = np.where(chrom_weights < max_weight)[0]
        chrom_cold_blocks = []
        block = []
        for i in range(where_cold.shape[0]-1):
            if where_cold[i] == where_cold[i+1] - 1:
                block += list(chrom_edges[where_cold[i]])
            else:
                chrom_cold_blocks.append(set(block))
                block = []
        cold_blocks += chrom_cold_blocks
    return cold_blocks

def find_constrained_blocks_losses(constrained_blocks, tree, genomes):
    losses = {cb_index:[] for cb_index in range(len(constrained_blocks))}
    for node in tree.get_nonterminals() + tree.get_terminals():
        if node != tree.root:
            blocks, GAB = synteny_from_simulations(genomes[tree.root.name],genomes[node.name])
            path_to_root = [node.name for node in tree.root.get_path(node.name)]
            for cb_index, cb in enumerate(constrained_blocks):
                common_nodes = len(set(losses[cb_index]).intersection(set(path_to_root)))
                if common_nodes == 0:
                    GAB_cb = GAB.subgraph(cb)
                    if len(list(ntx.connected_components(GAB_cb))) > 1:
                        losses[cb_index].append(node.name) 
    return losses

def synteny_from_simulations(GA, GB):
    shared_edges = set(GA.edges()).intersection(set(GB.edges()))
    GAB = ntx.Graph(shared_edges)
    synteny_blocks = list(ntx.connected_components(GAB))
    return synteny_blocks, GAB
