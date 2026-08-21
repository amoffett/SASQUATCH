import pickle as pkl
import numpy as np
from scipy import linalg, stats, optimize
from Bio import Phylo
from multiprocessing import Pool
from functools import partial
import warnings
from ._model_tools import *

def p_SBL(n, t_path, gamma_path, theta_path):
    n_branches = len(t_path)
    G = []
    O1 = []
    O2 = []
    for bi in range(n_branches):
        t = t_path[bi]
        gamma = gamma_path[bi]
        theta = theta_path[bi]
        G.append(gamma * t)
        O1.append(gamma * t * np.exp(-theta*(n-1)))
        O2.append(gamma * t * np.exp(-theta*n))
    G = np.sum(G)
    O1 = np.sum(O1)
    O2 = np.sum(O2)
    return (np.exp(-G*n+O1) - np.exp(-G*(n+1)+O2) - np.exp(-G*(n+1))*(1-np.exp(-G)))/(1-np.exp(-2*G))

def p_synteny_block_size(spA, spB, params, tree, branch_lengths, n_min = 3, n_max = 2000):
    path = get_path(spA,spB,tree)
    t_path = []
    gamma_path = []
    theta_path = []
    for branch in path:
        gamma, theta = params[branch]
        t = branch_lengths[branch]
        t_path.append(t)
        gamma_path.append(gamma)
        theta_path.append(theta)
    
    P_unnormed = np.array([p_SBL(n, t_path, gamma_path, theta_path) for n in np.arange(n_min,n_max+1)])
    Z = 1 - np.sum([p_SBL(n, t_path, gamma_path, theta_path) for n in np.arange(1,n_min)])
    P = P_unnormed / Z
    return P, path

def p_synteny_block_size_gradient(spA, spB, params, tree, branch_lengths, map_branch_to_param_indices, n_min = 3, n_max = 2000):
    path = get_path(spA,spB,tree)
    t_path = []
    gamma_path = []
    theta_path = []
    for branch in path:
        gamma, theta = params[branch]
        t = branch_lengths[branch]
        t_path.append(t)
        gamma_path.append(gamma)
        theta_path.append(theta)
    t_path = np.array(t_path).reshape([len(path),1])
    gamma_path = np.array(gamma_path).reshape([len(path),1])
    theta_path = np.array(theta_path).reshape([len(path),1])
    n_vals = np.vstack(len(path)*[np.arange(1,n_max+1)])
    
    G_AB = np.sum(gamma_path*t_path)
    O_AB_n = np.sum(gamma_path*t_path*np.exp(-theta_path*n_vals),axis=0)
    O_AB_nm1 = np.sum(gamma_path*t_path*np.exp(-theta_path*(n_vals-1)),axis=0)
    
    P_unnormed = np.array([p_SBL(n, t_path.flatten(), gamma_path.flatten(), theta_path.flatten()) for n in np.arange(n_min,n_max+1)])
    Z = 1 - np.sum([p_SBL(n, t_path.flatten(), gamma_path.flatten(), theta_path.flatten()) for n in np.arange(1,n_min)])
    P = P_unnormed / Z

    A = np.exp(-G_AB*n_vals[0])/(1-np.exp(-2*G_AB))
    dA_dgamma = t_path*((n_vals-2)*np.exp(-G_AB*(n_vals+2))-n_vals*np.exp(-G_AB*n_vals))/(1-np.exp(-2*G_AB))**2
    dA_dtheta = 0
    
    B = np.exp(O_AB_nm1)
    dB_dgamma = t_path*np.exp(-theta_path*(n_vals-1))*np.exp(O_AB_nm1)
    dB_dtheta = -gamma_path*t_path*(n_vals-1)*np.exp(-theta_path*(n_vals-1))*np.exp(O_AB_nm1)
    
    C = np.exp(O_AB_n - G_AB)
    dC_dgamma = t_path*(np.exp(-theta_path*n_vals)-1)*np.exp(O_AB_n-G_AB)
    dC_dtheta = -gamma_path*t_path*n_vals*np.exp(-theta_path*n_vals)*np.exp(O_AB_n-G_AB)
    
    D = np.exp(-G_AB)
    dD_dgamma = -t_path*np.exp(-G_AB)
    dD_dtheta = 0
    
    E = np.exp(-2*G_AB)
    dE_dgamma = -2*t_path*np.exp(-2*G_AB)
    dE_dtheta = 0
    
    dP_unnormed_dgamma = A * (dB_dgamma - dC_dgamma - dD_dgamma + dE_dgamma) + dA_dgamma * (B - C - D + E)
    dP_unnormed_dtheta = A * (dB_dtheta - dC_dtheta - dD_dtheta + dE_dtheta) + dA_dtheta * (B - C - D + E)
    
    dZ_dgamma = - np.sum(dP_unnormed_dgamma[:,:(n_min-1)],axis=1,keepdims=True)
    dZ_dtheta = - np.sum(dP_unnormed_dtheta[:,:(n_min-1)],axis=1,keepdims=True)
    
    P_grad_gamma = dP_unnormed_dgamma[:,(n_min-1):] / Z - np.vstack([len(path)*[P_unnormed]]) * dZ_dgamma / Z**2
    P_grad_theta = dP_unnormed_dtheta[:,(n_min-1):] / Z - np.vstack([len(path)*[P_unnormed]]) * dZ_dtheta / Z**2
        
    P_grad = np.zeros([2*len(params.keys()),n_vals.shape[1]-(n_min-1)])
    for nb, branch in enumerate(path):
        branch_gamma_index, branch_theta_index = map_branch_to_param_indices[branch]
        P_grad[branch_gamma_index] = P_grad_gamma[nb]
        P_grad[branch_theta_index] = P_grad_theta[nb]
    return P, P_grad, path

def KL_divergence(P, Q):
    where_non_zero = np.where((P != 0) * (Q != 0))[0]
    P = P[where_non_zero]
    Q = Q[where_non_zero]
    return np.nan_to_num(P * np.log(P / Q)).sum()

def KL_divergence_grad(P, Q, dQ):
    Q_fix = np.copy(Q)
    Q_fix[Q_fix == 0] = np.inf
    KL_gradient = -np.nan_to_num(P * (dQ / Q_fix)).sum(1)
    KL = KL_divergence(P, Q)
    return KL, KL_gradient

def F_AB(species, params_dict, map_branch_to_param_indices, tree, all_branches, branch_lengths, data, n_min = 3):
    spA, spB = species
    n_empirical, counts_empirical = fetch_data(spA,spB,data)
    n_max = n_empirical.max()
    p_empirical = counts_empirical[n_empirical >= n_min] / counts_empirical[n_empirical >= n_min].sum()
    n_empirical = n_empirical[n_empirical >= n_min]
    p_empirical_array = np.zeros(n_max-n_min+1)
    p_empirical_array[n_empirical-n_min] = p_empirical
    p_theory, path = p_synteny_block_size(spA,spB,params_dict,tree,branch_lengths,n_min=n_min,n_max=n_max)
    KL = KL_divergence(p_empirical_array,p_theory)

    return KL

def F_AB_gradient(spA, spB, params_dict, map_branch_to_param_indices, tree, all_branches, branch_lengths, data, n_min = 3, branch_dependent = True):
    n_empirical, counts_empirical = fetch_data(spA,spB,data)
    n_max = n_empirical.max()
    p_empirical = counts_empirical[n_empirical >= n_min] / counts_empirical[n_empirical >= n_min].sum()
    n_empirical = n_empirical[n_empirical >= n_min]
    p_empirical_array = np.zeros(n_max-n_min+1)
    p_empirical_array[n_empirical-n_min] = p_empirical
    p_theory, p_theory_grad, path = p_synteny_block_size_gradient(spA,spB,params_dict,tree,branch_lengths,map_branch_to_param_indices,n_max=n_max)

    KL, KL_grad = KL_divergence_grad(p_empirical_array,p_theory,p_theory_grad)
    if branch_dependent:
        return KL, KL_grad
    else:
        KL_grad_branch_ind = np.array([KL_grad[::2].sum(),KL_grad[1::2].sum()])
        return KL, KL_grad_branch_ind

def F_all_pairs(params, tree, all_branches, branch_lengths, data, n_min = 3, regularization = False):
    map_branch_to_param_indices = {all_branches[n]:[2*n,2*n+1] for n in range(len(all_branches))}
    params_dict = {}
    for nb, branch in enumerate(all_branches):
        params_dict[branch] = params[2*nb:2*(nb+1)]

    terms = [term.name for term in tree.get_terminals()]
    n_species = len(terms)
    sp_pairs = []
    for nA in range(n_species):
        for nB in range(nA+1,n_species):
            sp_pairs.append((terms[nA],terms[nB]))
    Z = n_species * (n_species - 1) / 2.
    F = 0
    for pair in sp_pairs:
        F += F_AB(pair,params_dict,map_branch_to_param_indices,tree,all_branches,branch_lengths,data,n_min=n_min)
    F = F / Z
    if regularization == True:
        gammas = params[::2]
        thetas = params[1::2]
        R = np.mean((gammas - gammas.mean())**2) + np.mean((thetas - thetas.mean())**2)
        F += R
    return F

def F_all_pairs_gradient(params, tree, all_branches, branch_lengths, data, n_min = 3, regularization = False, branch_dependent = True, verbose = False):
    map_branch_to_param_indices = {all_branches[n]:[2*n,2*n+1] for n in range(len(all_branches))}
    params_dict = {}
    for nb, branch in enumerate(all_branches):
        params_dict[branch] = params[2*nb:2*(nb+1)]

    terms = [term.name for term in tree.get_terminals()]
    n_species = len(terms)
    F = 0
    F_grad = 0
    for nA in range(n_species):
        for nB in range(nA+1,n_species):
            F_ab, F_ab_grad = F_AB_gradient(terms[nA],terms[nB],params_dict,map_branch_to_param_indices,tree,all_branches,branch_lengths,data,n_min=n_min,branch_dependent=branch_dependent)
            F += F_ab
            F_grad += F_ab_grad
    Z = n_species * (n_species - 1) / 2.
    F = F / Z
    F_grad = F_grad / Z
    if regularization == True:
        gammas = params[::2]
        thetas = params[1::2]
        R = np.mean((gammas - gammas.mean())**2) + np.mean((thetas - thetas.mean())**2)
        dR = np.zeros(params.shape[0])
        dR[::2] = (2 / gammas.shape[0]) * (gammas - gammas.mean())
        dR[1::2] = (2 / thetas.shape[0]) * (thetas - thetas.mean())
        F += R
        F_grad += dR
    if verbose == True:
        print(F,flush=True)
    return F, F_grad

def model_fit_minimization_step_mp(x0, tree, all_branches, branch_lengths, synteny_distributions, gamma_bounds, theta_bounds, max_steps, branch_dependent = True, regularization = False):
    if branch_dependent:
        if x0.shape[0] == 2 * len(all_branches):
            minimization_bounds = int(x0.shape[0]/2.) * [gamma_bounds, theta_bounds]
            f_f_gradient = lambda x: F_all_pairs_gradient(x, tree, all_branches, branch_lengths, synteny_distributions, regularization = regularization, branch_dependent = branch_dependent)
        else:
            raise ValueError("If the model is branch dependent, the number of entries in x0 should be 2 times the number of branches.")
    else:
        if x0.shape[0] == 2:
            minimization_bounds = [gamma_bounds, theta_bounds]
            f_f_gradient = lambda x: F_all_pairs_gradient(convert_two_parameters_to_full(x, len(all_branches)), tree, all_branches, branch_lengths, synteny_distributions, regularization = regularization, branch_dependent = branch_dependent)
        else:
            raise ValueError("If the model is not branch dependent, the number of entries in x0 should be 2.")
    opt = optimize.minimize(f_f_gradient,x0,method='SLSQP',jac=True,bounds=minimization_bounds,options={'maxiter':max_steps})
    return opt.x, opt.fun

def minimize_F(n_proc, n_initial_conditions, tree, all_branches, branch_lengths, synteny_distributions, gamma_bounds, theta_bounds, max_steps, branch_dependent = True):
    rng = np.random.default_rng()
    X0 = []
    print('Generating initial parameters',flush=True)
    for i in range(n_initial_conditions):
        while True:
            if branch_dependent:
                x0 = np.zeros(2*len(all_branches))
                gamma0 = rng.random(size=len(all_branches)) * (gamma_bounds[1] - gamma_bounds[0]) + gamma_bounds[0]
                theta0 = rng.random(size=len(all_branches)) * (theta_bounds[1] - theta_bounds[0]) + theta_bounds[0]
                x0[::2] = gamma0
                x0[1::2] = theta0
                F0 = F_all_pairs(x0, tree, all_branches, branch_lengths, synteny_distributions)
            else:
                x0 = np.zeros(2)
                gamma0 = rng.random() * (gamma_bounds[1] - gamma_bounds[0]) + gamma_bounds[0]
                theta0 = rng.random() * (theta_bounds[1] - theta_bounds[0]) + theta_bounds[0]
                x0[0] = gamma0
                x0[1] = theta0
                F0 = F_all_pairs(convert_two_parameters_to_full(x0, len(all_branches)), tree, all_branches, branch_lengths, synteny_distributions)
            if F0 < 100:
                break
        X0.append(np.copy(x0))

    print('Beginning minimization',flush=True)
    with Pool(processes=n_proc) as pool:
        opt_func = partial(model_fit_minimization_step_mp,tree=tree,all_branches=all_branches,branch_lengths=branch_lengths,synteny_distributions=synteny_distributions,gamma_bounds=gamma_bounds,theta_bounds=theta_bounds,max_steps=max_steps,branch_dependent=branch_dependent)
        pool_results = pool.imap(opt_func,X0)
        pool.close()
        pool.join()
    x_results = []
    F_results = []
    for pool_result in pool_results:
        x_results.append(pool_result[0])
        F_results.append(pool_result[1])
    x_results = np.vstack(x_results)
    return x_results, F_results
