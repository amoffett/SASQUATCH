import numpy as np
from scipy import stats
import networkx as ntx
from SASQUATCH.model import *

def simulate_genome_evolution_2cut(karyotype, branch_lengths, branch_params):
    if np.unique(karyotype).shape[0] != 1:
        raise ValueError("All chromosomes must be the same size.")
    
    n_branches = branch_lengths.shape[0]
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
    
    results = {}    
    T = np.sum(branch_lengths)
    t = 0
    nt = 0
    final_step = False
    while True:
        gamma = branch_params[nt,0]
        theta = branch_params[nt,1]
        gamma_array = np.array(edge_counter*[gamma])
        
        dt_vals = stats.expon.rvs(scale=1/gamma_array)
        dt_loc = dt_vals.argmin()
        dt = dt_vals.min()
        while dt > (np.sum(branch_lengths[:(nt+1)]) - t):
            if (t + dt >= T) and (nt == n_branches - 1):
                final_step = True
                dt = T - t
                t = T
                break
            else:
                t += (np.sum(branch_lengths[:(nt+1)]) - t)
            nt += 1
            gamma = branch_params[nt,0]
            theta = branch_params[nt,1]
            gamma_array = np.array(edge_counter*[gamma])
            dt_vals = stats.expon.rvs(scale=1/gamma_array)
            dt_loc = dt_vals.argmin()
            dt = dt_vals.min()
        t += dt
        
        if final_step:
            break
        
        initial_edge = tuple(edges[dt_loc])
        if initial_edge in edge_list:
            edge_list.remove(initial_edge)
        cut_size = stats.geom.rvs(1-np.exp(-theta))
        left_hand_initial_gene = edges[dt_loc,0]
        if int((left_hand_initial_gene + cut_size)/chrom_size) != int(left_hand_initial_gene/chrom_size):
            cut_size = 0
        final_edge = tuple(edges[int(dt_loc+cut_size)])
        if final_edge in edge_list:
            edge_list.remove(final_edge)
        
    G = ntx.Graph(edge_list)
    return G, G0

def survival_probability_branch(n, t, gamma, theta):
    z = (np.exp(-theta) - np.exp(-theta*(n-1)))/(1-np.exp(-theta))
    return np.exp(-gamma*t*(n+z))

def p_survival_simulations_theory(n, branch_lengths, branch_params):
    p_survive_theory = 1
    for i in range(len(branch_lengths)):
        t = branch_lengths[i]
        gamma, theta = branch_params[i]
        p = survival_probability_branch(n, t, gamma, theta)
        p_survive_theory *= p
    return p_survive_theory

def analyze_simulation(G, G0, n_max = 100):
    CC = [list(np.sort(list(cc)).astype(int)) for cc in ntx.connected_components(G)]
    singletons = G0.number_of_nodes() - G.number_of_nodes()
    h_cc = np.unique(singletons * [1] + [len(cc) for cc in CC],return_counts=True) 
    
    CC0 = [list(np.sort(list(cc0)).astype(int)) for cc0 in ntx.connected_components(G0)]
    CC0_lens = [len(cc0) for cc0 in CC0]
    p_survival_n = []
    for n in range(3,n_max+1):
        chunks0 = []
        for cc0_index in range(len(CC0)):
            n_chunks = int(np.floor(CC0_lens[cc0_index]/n))
            for n_chunk in range(n_chunks):
                chunk_genes = CC0[cc0_index][n_chunk*n:(n_chunk+1)*n]
                chunks0.append(','.join(np.array(chunk_genes).astype(str)))
        cc_strings = '-'
        for cc in CC:
            cc_strings += ','.join(np.array(cc).astype(str))+'-'
        present = 0
        for chunk0 in chunks0:
            if chunk0 in cc_strings:
                present += 1
        p_survival_n.append([n,present/len(chunks0)])
    p_survival_n = np.vstack(p_survival_n)
    
    return h_cc, p_survival_n

def p_synteny_block_size_simulations_theory(n_array, t_path, gamma_path, theta_path):
    Z = 1 - np.sum([p_SBL(n, t_path, gamma_path, theta_path) for n in range(1,n_array.min())])
    return np.array([p_SBL(n, t_path, gamma_path, theta_path) for n in n_array]) / Z
