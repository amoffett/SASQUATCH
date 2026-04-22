import numpy as np
from scipy import linalg, stats

def KL_divergence(b_hat, b, min_synteny = 3):
    b_hat = b_hat[(min_synteny-1):]
    b = b[(min_synteny-1):]
    p_hat = b_hat / np.sum(b_hat)
    p = b / np.sum(b)
    D = np.nan_to_num(p_hat * np.log(p_hat / p)).sum()
    #D = - np.nan_to_num(p_hat * np.log(p)).sum() - np.nan_to_num((1-p_hat) * np.log(1-p)).sum()
    return D

def dKL_divergence(b_hat, b, db_dkappa, db_dgamma, db_dtheta, min_synteny = 3, reduced = False):
    b_hat = b_hat[(min_synteny-1):]
    b = b[(min_synteny-1):]
    p_hat = b_hat / np.sum(b_hat)
    z = np.sum(b)
    p = b / z

    db_dkappa = db_dkappa[(min_synteny-1):]
    #dD_dkappa = - np.nan_to_num(((p_hat / p) - (1 - p_hat) / (1 - p)) * (db_dkappa - p * np.sum(db_dkappa)) / z).sum()
    dD_dkappa = - np.nan_to_num((p_hat / p) * (db_dkappa - p * np.sum(db_dkappa)) / z).sum()
    if reduced == False:
        db_dgamma = db_dgamma[(min_synteny-1):]
        #dD_dgamma = - np.nan_to_num(((p_hat / p) - (1 - p_hat) / (1 - p)) * (db_dgamma - p * np.sum(db_dgamma)) / z).sum()
        dD_dgamma = - np.nan_to_num((p_hat / p) * (db_dgamma - p * np.sum(db_dgamma)) / z).sum() 

        db_dtheta = db_dtheta[(min_synteny-1):]
        #dD_dtheta = - np.nan_to_num(((p_hat / p) - (1 - p_hat) / (1 - p)) * (db_dtheta - p * np.sum(db_dtheta)) / z).sum()
        dD_dtheta = - np.nan_to_num((p_hat / p) * (db_dtheta - p * np.sum(db_dtheta)) / z).sum()
    else:
        dD_dgamma = 0
        dD_dtheta = 0
    return dD_dkappa, dD_dgamma, dD_dtheta

def prepare_b_empirical(N, b_empirical, species_list, min_synteny = 3):
    b_empirical_fixed = {}
    for spA in species_list:
        for spB in species_list:
            if spA != spB:
                n, b = b_empirical[(spA,spB)]
                b_hat = np.zeros(N)
                b_hat[n[n >= min_synteny]-1] = b[n >= min_synteny]
                b_empirical_fixed[(spA,spB)] = b_hat
    return b_empirical_fixed

def calculate_F(N, b_empirical, all_branches, species_list, branch_solutions, b0, tree, min_synteny = 3):
    F = 0
    Z = len(species_list)**2 - len(species_list)
    for spA in species_list:
        for spB in species_list:
            if spA != spB:
                b_hat = b_empirical[(spA,spB)]
                b = b_theory(N, spA, spB, branch_solutions, b0, tree)
                D = KL_divergence(N, b_hat, b, min_synteny = min_synteny)
                F += D
    F = F / Z
    return F

def calculate_dF(N, b_empirical, all_branches, species_list, branch_solutions, branch_solutions_dkappa, branch_solutions_dgamma, branch_solutions_dtheta, b0, tree, min_synteny = 3):
    dF = {}
    Z = len(species_list)**2 - len(species_list)
    for spA in species_list:
        for spB in species_list:
            if spA != spB:
                dF[(spA,spB)] = {}
                b = b_theory(N, spA, spB, branch_solutions, b0, tree)
                b_hat = b_empirical[(spA, spB)]
                db_dkappa, db_dgamma, db_dtheta = db_theory(N, spA, spB, branch_solutions, branch_solutions_dkappa, branch_solutions_dgamma, branch_solutions_dtheta, b0, tree)
                path = get_path(spA,spB,tree)
                for branch in path:
                    if branch in db_dkappa.keys():
                        dF[(spA,spB)][branch] = dKL_divergence(N, b_hat, b, db_dkappa[branch], db_dgamma[branch], db_dtheta[branch], min_synteny = min_synteny)
                    else:
                        dF[(spA,spB)][branch] = np.zeros(3)
    dF_array = []
    for branch in all_branches:
        dF_branch = np.zeros(3)
        for key in dF.keys():
            if branch in dF[key].keys():
                dF_branch += dF[key][branch]
        dF_array += list(dF_branch)
    dF_array = np.array(dF_array) / Z
    return dF_array

def calculate_F_dF(params, N, b_empirical, b0, tree, all_branches, species_list, branch_lengths, cut_distribution, min_synteny = 3, nproc = 4):
    branch_solutions, branch_solutions_dkappa, branch_solutions_dgamma, branch_solutions_dtheta = generate_branch_solutions(N, all_branches, params, branch_lengths, cut_distribution, nproc = nproc)

    F = calculate_F(N, b_empirical, all_branches, species_list, branch_solutions, b0, tree, min_synteny = min_synteny)
    dF = calculate_dF(N, b_empirical, all_branches, species_list, branch_solutions, branch_solutions_dkappa, branch_solutions_dgamma, branch_solutions_dtheta, b0, tree, min_synteny = min_synteny)
    del branch_solutions, branch_solutions_dkappa, branch_solutions_dgamma, branch_solutions_dtheta
    return F, dF
