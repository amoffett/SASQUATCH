import numpy as np
import pymbar
from SASQUATCH.model import *
from ._mcmc_sampling import *	
from ._phylogeny_tools import *

def compute_MBAR_expectations(block_trait_data, mcmc_sample_dictionary, subsampling_factor = 100):
    species_labels = block_trait_data['column_labels']
    block_sizes = np.unique([block_trait_data['data'][key]['block_size'] for key in block_trait_data['data'].keys()])
    p_values = []
    for block_size in block_sizes:
        print("Working on n = %i"%block_size)
        mcmc_samples = mcmc_sample_dictionary[block_size]
        n_replicas = mcmc_samples['betas'].shape[0]
        n_samples = mcmc_samples['potentials'][:,::subsampling_factor].shape[1]
        u_kn = np.vstack([mcmc_samples['betas'][i] * mcmc_samples['potentials'][:,::subsampling_factor].flatten() for i in range(n_replicas)])
        N_k = n_replicas * [n_samples]
        print("Running MBAR")
        mbar = pymbar.MBAR(u_kn,N_k,solver_protocol="robust")
        T_n = mcmc_samples['conservation times'][:,::subsampling_factor].flatten()
        nlog_q_n = mcmc_samples['potentials'][:,::subsampling_factor].flatten()
        T_mean = mbar.compute_expectations(T_n)['mu'][0]
        block_size_indices = [index for index in block_trait_data['data'].keys() if block_trait_data['data'][index]['block_size'] == block_size]
        i = 1
        for index in block_size_indices:
            print("Working on %i out of %i"%(i,len(block_size_indices)))
            H = camin_sokal_parsimony(block_trait_data['data'][index]['trait_array'], species_labels, tree)
            T = total_time_conserved(H, tree)
            nlog_q = - np.log(calculate_qHn(block_size, H, params, tree))
            O_T = (T_n >= T).astype(int)
            O_q = (nlog_q_n >= nlog_q).astype(int)
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

            p_values.append([index, T, T_mean, p_T, sig_T, p_q, sig_q])
            i += 1
    p_values = np.vstack(p_values)
    p_values = p_values[np.argsort(p_values[:,0])]
