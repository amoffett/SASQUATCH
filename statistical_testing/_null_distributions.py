import numpy as np
from GenomicTools.tools import *
from ._phylogeny_tools import *

def survival_probability_branch(n, t, gamma, theta):
    z = (np.exp(-theta) - np.exp(-theta*(n-1)))/(1-np.exp(-theta))
    return np.exp(-gamma*t*(n+z))

def calculate_Pa(n, a, tree, params, branch_lengths, species_labels):
    a = a.astype(int)
    p_l = {}
    for n_term, a_term in enumerate(a):
        p_l[species_labels[n_term]] = np.zeros(2)
        p_l[species_labels[n_term]][a_term] = 1.
    terminals = tree.get_terminals()

    for node in tree.get_nonterminals(order='postorder'):
        r_node = np.zeros(2)

        t1 = branch_lengths[node[0].name]
        gamma1, theta1 = params[node[0].name]
        s1 = survival_probability_branch(n,t1,gamma1,theta1)
        R1 = np.array([[1,0],[1-s1,s1]])

        t2 = branch_lengths[node[1].name]
        gamma2, theta2 = params[node[1].name]
        s2 = survival_probability_branch(n,t2,gamma2,theta2)
        R2 = np.array([[1,0],[1-s2,s2]])

        r0A = R1[0,0] * p_l[node[0].name][0] * R2[0,0] * p_l[node[1].name][0]
        r0B = R1[0,1] * p_l[node[0].name][1] * R2[0,0] * p_l[node[1].name][0]
        r0C = R1[0,0] * p_l[node[0].name][0] * R2[0,1] * p_l[node[1].name][1]
        r0D = R1[0,1] * p_l[node[0].name][1] * R2[0,1] * p_l[node[1].name][1]
        r0 = r0A + r0B + r0C + r0D 

        r1A = R1[1,0] * p_l[node[0].name][0] * R2[1,0] * p_l[node[1].name][0]
        r1B = R1[1,1] * p_l[node[0].name][1] * R2[1,0] * p_l[node[1].name][0]
        r1C = R1[1,0] * p_l[node[0].name][0] * R2[1,1] * p_l[node[1].name][1]
        r1D = R1[1,1] * p_l[node[0].name][1] * R2[1,1] * p_l[node[1].name][1]
        r1 = r1A + r1B + r1C + r1D

        p_l[node.name] = np.array([r0,r1])

    return p_l[tree.root.name][1]

def calculate_approximate_T_distribution(n, tree, params, dt, species_present_conditional = [], species_absent_conditional = []):
    if len(set(species_present_conditional) & set(species_absent_conditional)) > 0:
        raise ValueError("The lists species_present_conditional and species_absent_conditional should be disjoint.")
    terminals = tree.get_terminals()
    T_max = tree.total_branch_length()
    bins = [0.]
    while bins[-1] < T_max:
        bins.append(bins[-1] + dt)
    bins.append(bins[-1] + dt)
    bins = np.array(bins)
    nbins = bins.shape[0] - 1
    T_vals = (bins[1:] + bins[:-1]) / 2.
    
    present_branches = []
    for sp in species_present_conditional:
        present_branches += [node.name for node in tree.root.get_path(sp)]
    present_branches = set(present_branches) 
        
    T_l = {}
    s_l = {}
    terminals = tree.get_terminals()
    all_nodes = tree.get_terminals() + tree.get_nonterminals()
    for node in all_nodes:
        if node != tree.root:
            t = branch_lengths[node.name]
            gamma, theta = params[node.name]
            s = survival_probability_branch(n,t,gamma,theta)
            s_l[node.name] = s
            b = np.digitize(t,bins) - 1
            tb = T_vals[b]
            if node in terminals:
                T_node = np.zeros(b+1)
                if node.name in species_present_conditional:
                    T_node[b] = 1.
                elif node.name in species_absent_conditional:
                    T_node[0] = 1.
                else:
                    T_node[0] = 1. - s
                    T_node[b] += s
                T_l[node.name] = T_node
        
    for nnode, node in enumerate(tree.get_nonterminals('postorder')):
        child1 = node[0]
        if child1 not in terminals:
            b_branch1 = np.digitize(child1.branch_length,bins) - 1
            b_child1 = T_l[child1.name].shape[0]
            T1 = np.zeros(b_branch1 + b_child1)
            pos_indices = np.where(T_l[child1.name] > 0)[0]
            if child1.name not in present_branches:
                T1[0] = 1 - s_l[child1.name]
                T1[pos_indices + b_branch1] = s_l[child1.name] * T_l[child1.name][pos_indices]
            else:
                T1[pos_indices + b_branch1] =  T_l[child1.name][pos_indices]
        else:
            T1 = T_l[child1.name]
        child2 = node[1]
        if child2 not in terminals:
            b_branch2 = np.digitize(child2.branch_length,bins) - 1
            b_child2 = T_l[child2.name].shape[0]
            T2 = np.zeros(b_branch2 + b_child2)
            pos_indices = np.where(T_l[child2.name] > 0)[0]
            if child2.name not in present_branches:
                T2[0] = 1 - s_l[child2.name]
                T2[pos_indices + b_branch2] = s_l[child2.name] * T_l[child2.name][pos_indices]
            else:
                T2[pos_indices + b_branch2] =  T_l[child2.name][pos_indices]
        else:
            T2 = T_l[child2.name]
            
        T_l[node.name] = np.convolve(T1,T2)
        
    p_T = T_l[tree.root.name]
    p_T = np.array(list(p_T) + list(np.zeros(T_vals.shape[0] - p_T.shape[0])))
    return T_vals, p_T

def sample_T_distribution(n, tree, params, n_sims, species_present_conditional = [], species_absent_conditional = []):
    present_conditioned_nodes = []
    for sp in species_present_conditional:
        present_conditioned_nodes += [node.name for node in tree.root.get_path(sp)]
    present_conditioned_nodes = np.unique(present_conditioned_nodes)
    
    T_sample = []
    i = 0
    while i < n_sims:
        node_list = tree.get_nonterminals('preorder')
        T = 0
        while True:
            if len(node_list) == 0:
                break
            node = node_list[0]
            child1 = node[0]
            child2 = node[1]
            
            gamma1, theta1 = params[child1.name]
            t1 = child1.branch_length
            gamma2, theta2 = params[child2.name]
            t2 = child2.branch_length

            s1 = survival_probability_branch(n, t1, gamma1, theta1)
            s2 = survival_probability_branch(n, t2, gamma2, theta2)
            r1 = np.random.rand()
            r2 = np.random.rand()
    
            if (child1.name in species_absent_conditional):
                pass
            elif (r1 <= s1) or (child1.name in present_conditioned_nodes):
                T += t1
            else:
                descendents1 = child1.get_nonterminals()
                for d1 in descendents1:
                    node_list.remove(d1)

            if (child2.name in species_absent_conditional):
                pass
            elif (r2 <= s2) or (child2.name in present_conditioned_nodes):
                T += t2
            else:
                descendents2 = child2.get_nonterminals()
                for d2 in descendents2:
                    node_list.remove(d2)
            node_list.remove(node)
        i += 1
        T_sample.append(T)
        print(i,end='\r',flush=True)

    return T_sample

def bin_T_distribution(T_vals, P_T, bins):
    P_T_binned = np.zeros(len(bins)-1)
    for nt, t in enumerate(T_vals):
        P_T_binned[np.digitize(t,bins)-1] += P_T[nt]
    T_bins = (bins[1:]-bins[:-1])/2.
    return T_bins, P_T_binned

