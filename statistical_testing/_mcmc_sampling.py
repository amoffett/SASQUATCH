import numpy as np
from GenomicTools.tools import *
from multiprocessing import Pool
from functools import partial
from ._phylogeny_tools import *

def survival_probability_branch(n, t, gamma, theta):
    z = (np.exp(-theta) - np.exp(-theta*(n-1)))/(1-np.exp(-theta))
    return np.exp(-gamma*t*(n+z))

def calculate_qHn(n, H, params, tree):
    all_nodes = tree.get_terminals() + tree.get_nonterminals()
    qH = 1
    for node in all_nodes:
        if node == tree.root:
            continue
        else:
            if (node == tree.root[0]) or (node == tree.root[1]):
                parent = tree.root
            else:
                parent = tree.root.get_path(node)[-2]
            if (H[parent.name] == 1) and (H[node.name] == 1):
                t_l = node.branch_length
                gamma_l, theta_l = params[node.name]
                s_l = survival_probability_branch(n, t_l, gamma_l, theta_l)
                qH *= s_l
            elif (H[parent.name] == 1) and (H[node.name] == 0):
                t_l = node.branch_length
                gamma_l, theta_l = params[node.name]
                s_l = survival_probability_branch(n, t_l, gamma_l, theta_l)
                qH *= (1 - s_l)
    return qH

def mcmc_step(X1, beta, n, species_labels, tree, params):
    S = X1.shape[0]
    s = np.random.choice(range(S))
    X2 = np.copy(X1)
    X2[s] = 1 - X2[s]
    H1, L1 = camin_sokal_parsimony(X1,species_labels,tree)
    H2, L2 = camin_sokal_parsimony(X2,species_labels,tree)
    q1 = calculate_qHn(n, H1, params, tree)
    q2 = calculate_qHn(n, H2, params, tree)
    A = np.min([1,(q2/q1)**beta])
    r = np.random.rand()
    if r < A:
        T2 = total_time_conserved(H2, tree)
        return np.array(X2), q2, T2
    else:
        T1 = total_time_conserved(H1, tree)
        return np.array(X1), q1, T1

def exchange_step(XA, betaA, XB, betaB, n, species_labels, tree, params):
    HA, LA = camin_sokal_parsimony(XA,species_labels,tree)
    HB, LB = camin_sokal_parsimony(XB,species_labels,tree)
    qA = calculate_qHn(n, HA, params, tree)
    qB = calculate_qHn(n, HB, params, tree)
    VA = -np.log(qA)
    VB = -np.log(qB)
    A = np.min([1,np.exp(-betaB*VA-betaA*VB)/np.exp(-betaA*VA-betaB*VB)])
    r = np.random.rand()
    if r < A:
        return np.array(XB), np.array(XA)
    else:
        return np.array(XA), np.array(XB)

def mcmc_replica(rX, n_steps, betas, n, species_labels, tree, params):
    r, X = rX
    beta = betas[r]
    n_step = 0
    q = np.zeros(n_steps)
    T = np.zeros(n_steps)
    while n_step < n_steps:
        X, qn, Tn = mcmc_step(X, beta, n, species_labels, tree, params)
        q[n_step] = qn
        T[n_step] = Tn
        n_step += 1
    return r, X, q, T

def sample_histories_mcmc_parallel_tempering(n, initial_trait_array, tree, species_labels, params, samples = 50000, burn_in = 1000, replicas = 10, steps_exchange = 500, beta_max = 1/3, save_trait_arrays = False, parallel = False, verbose = False):
    X = {i:np.copy(initial_trait_array) for i in range(replicas)}
    q = np.zeros([replicas,samples])
    T = np.zeros([replicas,samples])
    if save_trait_arrays:
        X_out = np.zeros([replicas, initial_trait_array.shape[0], int(samples/steps_exchange)])

    S = initial_trait_array.shape[0]
    beta_rate = np.log(1/beta_max) / (replicas-1)
    betas = 1/np.exp(beta_rate*np.arange(replicas))

    if verbose:
        print("Starting burn in",flush=True)
    i = 1
    if not parallel:
        while i <= burn_in:
            for r in X.keys():
                X2, q2, T2 = mcmc_step(X[r], betas[r], n, species_labels, tree, params)
                X[r] = X2
            i += 1
    else:
        rX_list = [[r,X[r]] for r in X.keys()]
        with Pool(processes=replicas) as pool:
            mcmc_replica_func = partial(mcmc_replica,n_steps=burn_in,betas=betas,n=n,species_labels=species_labels,tree=tree,params=params)
            pool_results = pool.imap(mcmc_replica_func,rX_list)
            pool.close()
            pool.join()
        for pool_result in pool_results:
            X[pool_result[0]] = pool_result[1]

    if verbose:
        print("Starting MCMC sampling",flush=True)
    exchange_mode = 0
    if not parallel:
        i = 1
        while i <= samples:
            if verbose:
                print("MCMC step %i"%i,flush=True)
            for r in X.keys():
                X2, q2, T2 = mcmc_step(X[r], betas[r], n, species_labels, tree, params)
                X[r] = X2
                q[r,i-1] = q2
                T[r,i-1] = T2
            if i % steps_exchange == 0:
                if save_trait_arrays:
                    for r in X.keys():
                        X_out[r,:,int((i-1)/steps_exchange)] = X[r]
                if verbose:
                    print("Exchange step %i"%np.round(i/steps_exchange),flush=True)
                if exchange_mode == 0:
                    for r_ex in np.arange(replicas-1)[::2]:
                        XA, XB = exchange_step(X[r_ex], betas[r_ex], X[r_ex+1], betas[r_ex+1], n, species_labels, tree, params)
                        X[r_ex] = XA
                        X[r_ex+1] = XB
                else:
                    for r_ex in np.arange(replicas-1)[1::2]:
                        XA, XB = exchange_step(X[r_ex], betas[r_ex], X[r_ex+1], betas[r_ex+1], n, species_labels, tree, params)
                        X[r_ex] = XA
                        X[r_ex+1] = XB
                exchange_mode = 1 - exchange_mode
            i += 1
    else:
        i = 1
        while i <= np.round(samples/steps_exchange):
            if verbose:
                print("MCMC batch %i"%i,flush=True)
            rX_list = [[r,X[r]] for r in X.keys()]
            with Pool(processes=replicas) as pool:
                mcmc_replica_func = partial(mcmc_replica,n_steps=steps_exchange,betas=betas,n=n,species_labels=species_labels,tree=tree,params=params)
                pool_results = pool.imap(mcmc_replica_func,rX_list)
                pool.close()
                pool.join()
            for pool_result in pool_results:
                X[pool_result[0]] = pool_result[1]
                q[pool_result[0],(i-1)*steps_exchange:i*steps_exchange] = pool_result[2]
                T[pool_result[0],(i-1)*steps_exchange:i*steps_exchange] = pool_result[3]

                if save_trait_arrays:
                    X_out[pool_result[0],:,int((i-1)/steps_exchange)] = pool_result[1]
            if verbose:
                print("Exchange step %i"%i,flush=True)
            if exchange_mode == 0:
                for r_ex in np.arange(replicas-1)[::2]:
                    XA, XB = exchange_step(X[r_ex], betas[r_ex], X[r_ex+1], betas[r_ex+1], n, species_labels, tree, params)
                    X[r_ex] = XA
                    X[r_ex+1] = XB
            else:
                for r_ex in np.arange(replicas-1)[1::2]:
                    XA, XB = exchange_step(X[r_ex], betas[r_ex], X[r_ex+1], betas[r_ex+1], n, species_labels, tree, params)
                    X[r_ex] = XA
                    X[r_ex+1] = XB
            exchange_mode = 1 - exchange_mode
            i += 1

    results = {}
    results['betas'] = betas
    results['potentials'] = - np.log(q)
    results['conservation times'] = T
    return results, X_out
