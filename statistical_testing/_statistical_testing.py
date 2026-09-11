import numpy as np
import pymbar
from multiprocessing import Pool
from functools import partial
from SASQUATCH.model import *
from ._mcmc_sampling import *	
from ._phylogeny_tools import *

def calculate_reweighted_potential(trait_array, n, params_block, species_labels, tree):
    H = camin_sokal_parsimony(trait_array, species_labels, tree)
    q = calculate_qHn(n, H, params_block, tree)
    return -np.log(q)

def reweight_potentials(block_index_name, trait_arrays, path_to_mcmc_trait_data, params, species_labels, tree):
    block_index, block_name = block_index_name
    n = trait_arrays['data'][block_name]['block_size']
    block_weight = trait_arrays['weights'][block_index]
    params_block = {}
    for key in params.keys():
        gamma = params[key][0] * block_weight
        theta = params[key][1]
        params_block[key] = [gamma, theta]
    X = np.load(path_to_mcmc_trait_data + 'trait_arrays_mcmc_n%i.npy'%n)
    n_replicas = X.shape[0]
    n_samples = X.shape[2]
    V = np.zeros([n_replicas,n_samples])
    for r in range(n_replicas):
        for s in range(n_samples):
            V_rs = calculate_weighted_potential(X[r,:,s], n, params_block, species_labels, tree)
            V[r,s] = V_rs
    del X
    return block_name, V

def reweight_potentials_parallel(n_processes, trait_arrays, path_to_mcmc_trait_data, params, species_labels, tree):
    V_blocks = {}
    block_indices_names = [(index,name) for index, name in enumerate(trait_arrays['data'].keys())]
    with Pool(processes=n_processes) as pool:
        reweight_func = partial(reweight_potentials,trait_arrays=trait_arrays,path_to_mcmc_trait_data=path_to_mcmc_trait_data,params=params,species_labels=species_labels,tree=tree)
        pool_results = pool.imap(reweight_func,block_indices_names)
        pool.close()
        pool.join()
    for pool_result in pool_results:
        V_blocks[pool_result[0]] = pool_result[1]
    return V_blocks

def compute_MBAR_expectations_block(n, block_trait_array, mcmc_sample, species_labels, params, tree, subsampling_factor = 1):
    p_values = []
    n_replicas = mcmc_sample['betas'].shape[0]
    n_samples = mcmc_sample['potentials'][:,::subsampling_factor].shape[1]
    u_kn = np.vstack([mcmc_sample['betas'][i] * mcmc_sample['potentials'][:,::subsampling_factor].flatten() for i in range(n_replicas)])
    N_k = n_replicas * [n_samples]
    mbar = pymbar.MBAR(u_kn,N_k,solver_protocol="robust")
    T_n = mcmc_sample['conservation times'][:,::subsampling_factor].flatten()
    V_n = mcmc_sample['potentials'][:,::subsampling_factor].flatten()
    T_mean = mbar.compute_expectations(T_n)['mu'][0]
    H = camin_sokal_parsimony(block_trait_array, species_labels, tree)
    T = total_time_conserved(H, tree)
    V = - np.log(calculate_qHn(n, H, params, tree))
    O_T = (T_n >= T).astype(int)
    O_q = (V_n >= V).astype(int)
    if np.all(O_T == 0):
        p_T = 0.0
        sig_T = 0.0
    elif np.all(O_T == 1):
        p_T = 1.0
        sig_T = 0.0
    else:
        E_T = mbar.compute_expectations(O_T)
        p_T = E_T['mu'][0]
        sig_T = E_T['sigma'][0]
    if np.all(O_q == 0):
        p_q = 0.0
        sig_q = 0.0
    elif np.all(O_q == 1):
        p_q = 1.0
        sig_q = 0.0
    else:
        E_q = mbar.compute_expectations(O_q)
        p_q = E_q['mu'][0]
        sig_q = E_q['sigma'][0]

    return T, T_mean, p_T, sig_T, p_q, sig_q
