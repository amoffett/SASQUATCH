import numpy as np
from scipy import linalg, stats, optimize
from Bio import Phylo
import networkx as ntx
from io import StringIO

def simulate_full_genome_evolution(G, weights, T, k):
    Gt = ntx.Graph(G)
    n_sites = Gt.number_of_edges()
    k = k * n_sites
    wt = {edge:weights[edge] for edge in weights.keys()}
    t = 0
    PE = []
    while t < T:
        dt = stats.expon.rvs(scale=1/k)
        if t + dt > T:
            break
        else:
            edge_list = list(wt.keys())
            weight_array = np.array([wt[edge] for edge in edge_list])
            p_edge = weight_array / weight_array.sum()
            swapped_edge_indices = np.random.choice(range(len(edge_list)),size=2,replace=False,p=p_edge)
            edge1 = edge_list[swapped_edge_indices[0]]
            edge2 = edge_list[swapped_edge_indices[1]]
            Gt.remove_edges_from([edge1,edge2])
            edge1_subgraph_nodesA = list(ntx.dfs_preorder_nodes(Gt,source=edge1[0]))
            edge1_subgraph_nodesB = list(ntx.dfs_preorder_nodes(Gt,source=edge1[1]))
            if edge2[0] in edge1_subgraph_nodesA:
                new_edge1 = (edge1[0],edge2[1])
                new_edge2 = (edge1[1],edge2[0]) 
            elif edge2[0] in edge1_subgraph_nodesB:
                new_edge1 = (edge1[0],edge2[0])
                new_edge2 = (edge1[1],edge2[1])
            elif edge2[1] in edge1_subgraph_nodesA:
                new_edge1 = (edge1[0],edge2[0])
                new_edge2 = (edge1[1],edge2[1]) 
            elif edge2[1] in edge1_subgraph_nodesB:
                new_edge1 = (edge1[1],edge2[0])
                new_edge2 = (edge1[0],edge2[1])
            else:
                edge_join_choice = np.random.randint(2)
                new_edge1 = (edge1[0],edge2[edge_join_choice])
                new_edge2 = (edge1[1],edge2[1-edge_join_choice])    
            Gt.add_edges_from([new_edge1,new_edge2])
            wt[new_edge1] = np.random.choice([wt[edge1], wt[edge2]]) # (wt[edge1] + wt[edge2]) / 2
            wt[new_edge2] = np.random.choice([wt[edge1], wt[edge2]]) # (wt[edge1] + wt[edge2]) / 2
            del wt[edge1], wt[edge2]
        t += dt
    return Gt

def simulate_genome_phylogeny(G0, weights, tree, rates):
    i = 0
    genomes = {}
    genomes[tree.root.name] = ntx.Graph(G0)
    for node in tree.get_nonterminals(order='preorder')[1:]:
        path_to_root = tree.root.get_path(node)
        if len(path_to_root) == 1:
            parent = tree.root.name
        else:
            parent = path_to_root[-2].name
        G_parent = genomes[parent]
        G = simulate_full_genome_evolution(G_parent, weights, node.branch_length, rates[node.name])
        genomes[node.name] = ntx.Graph(G)

        i += 1
        print(i,end='\r',flush=True)

    for node in tree.get_terminals():
        path_to_root = tree.root.get_path(node)
        if len(path_to_root) == 1:
            parent = tree.root.name
        else:
            parent = path_to_root[-2].name
        G_parent = genomes[parent]
        G = simulate_full_genome_evolution(G_parent, weights, node.branch_length, rates[node.name])
        genomes[node.name] = ntx.Graph(G)   

        i += 1
        print(i,end='\r',flush=True)
        
    return genomes
