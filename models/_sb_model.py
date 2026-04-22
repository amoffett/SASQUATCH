import numpy as np
from scipy import linalg, stats
from multiprocessing import Pool
from functools import partial

def R_matrix(N):
    R = 2 * np.triu(np.ones(N))
    np.fill_diagonal(R,-np.arange(N))
    return R

def R_V_matrix(N):
    V = np.identity(N)
    c = np.ones(N)
    d = -2 * c
    V = V + np.diag(d,1)[:N,:N] + np.diag(c,2)[:N,:N]
    return V

def R_V_inv_matrix(N):
    V_inv = np.zeros([N,N])
    for i in range(N):
        V_inv[i,i:] = np.arange(1,N+1-i)
    return V_inv

def R_L_matrix(N):
    L = np.zeros([N,N])
    np.fill_diagonal(L,-np.arange(N))
    return L

def S_dS_matrices(N, theta, dtheta, cut_distribution):
    if cut_distribution == 'zipf':
        rho = lambda x: stats.zipf.pmf(x,theta+1)
        sum_rho = lambda x: stats.zipf.cdf(x,theta+1)
        Drho = lambda x: stats.zipf.pmf(x,theta+1+dtheta)
        sum_Drho = lambda x: stats.zipf.cdf(x,theta+1+dtheta)
    elif cut_distribution == 'geometric':
        q = 1 - np.exp(-theta)
        Dq = 1 - np.exp(-(theta+dtheta))
        rho = lambda x: stats.geom.pmf(x,q)
        sum_rho = lambda x: stats.geom.cdf(x,q)
        Drho = lambda x: stats.geom.pmf(x,Dq)
        sum_Drho = lambda x: stats.geom.cdf(x,Dq)
    else:
        raise ValueError("'cut_distribution' must be either 'zipf' or 'geometric'.")

    J = np.ones([N,N]) * np.arange(N)
    I = J.T

    sum_rho_array = sum_rho((J-I)[0])
    JI = np.zeros([N,N])
    for i in range(N):
        JI[i,(i+1):] = sum_rho_array[1:(N-i)]
    S = np.triu((J-I+1)*rho(I+1)+2*JI,1)
    S_diag = (np.cumsum(np.arange(N)*rho(np.arange(N)))-(np.arange(1,N+1)+1)*sum_rho(np.arange(N)))
    np.fill_diagonal(S,S_diag)

    sum_Drho_array = sum_Drho((J-I)[0])
    DJI = np.zeros([N,N])
    for i in range(N):
        DJI[i,(i+1):] = sum_Drho_array[1:(N-i)]
    DS = np.triu((J-I+1)*Drho(I+1)+2*DJI,1)
    DS_diag = (np.cumsum(np.arange(N)*Drho(np.arange(N)))-(np.arange(1,N+1)+1)*sum_Drho(np.arange(N)))
    np.fill_diagonal(DS,DS_diag)

    dS_dtheta = (DS - S) / dtheta
    return S, dS_dtheta

def Q_dQ_matrix(N, kappa, gamma, theta, dtheta, cut_distribution, reduced = False):
    R = R_matrix(N)
    if reduced == False:
        S, dS_dtheta = S_dS_matrices(N, theta, dtheta, cut_distribution)
        Q = kappa * R + gamma * S
        dQ_dkappa = R
        dQ_dgamma = S
        dQ_dtheta = gamma * dS_dtheta
    else:
        Q = kappa * R
        dQ_dkappa = R
        dQ_dgamma = 0 * R
        dQ_dgamma = 0 * R
    return Q, dQ_dkappa, dQ_dgamma, dQ_dtheta

def branch_solution(branch, N, params, branch_lengths, dtheta, cut_distribution, reduced = False):
    kappa, gamma, theta = params[branch]
    t = branch_lengths[branch]
    Q, dQ_dkappa, dQ_dgamma, dQ_dtheta = Q_dQ_matrix(N, kappa, gamma, theta, dtheta, cut_distribution, reduced = reduced)

    solution = linalg.expm(Q*t)
    solution_dkappa = (dQ_dkappa*t).dot(solution)
    solution_dgamma = (dQ_dgamma*t).dot(solution)
    solution_dtheta = (dQ_dtheta*t).dot(solution)

    return branch, solution, solution_dkappa, solution_dgamma, solution_dtheta

def generate_branch_solutions(N, all_branches, params, branch_lengths, cut_distribution, dtheta = 1e-8, reduced = False, nproc = None):
    if type(params) == np.ndarray:
        if params.shape[0] == len(all_branches) * 3:
            PAR = np.copy(params)
            params = {}
            for n, branch in enumerate(all_branches):
                params[branch] = PAR[3*n:3*(n+1)]
        else:
            raise ValueError("'params' is the wrong size.")
    elif type(params) == dict:
        pass
    else:
        raise ValueError("'params' must be a numpy array or dictionary.")

    results = {}
    with Pool(processes=nproc) as pool:
        branch_sol = partial(branch_solution,N=N,params=params,branch_lengths=branch_lengths,dtheta=dtheta,cut_distribution=cut_distribution,reduced=reduced)
        results = pool.map(branch_sol,all_branches)
        pool.close()
        pool.join()

    branch_solutions = {}
    branch_solutions_dkappa = {}
    branch_solutions_dgamma = {}
    branch_solutions_dtheta = {}
    for result in results:
        branch_solutions[result[0]] = result[1]
        branch_solutions_dkappa[result[0]] = result[2]
        branch_solutions_dgamma[result[0]] = result[3]
        branch_solutions_dtheta[result[0]] = result[4]
    return branch_solutions, branch_solutions_dkappa, branch_solutions_dgamma, branch_solutions_dtheta

def b_theory(N, spA, spB, branch_solutions, b0, tree):
    path = get_path(spA,spB,tree)
    b_AB = b0[path[0]]
    for branch in path:
        b_AB = branch_solutions[branch].dot(b_AB)
    return b_AB

def db_theory(N, spA, spB, branch_solutions, branch_solutions_dkappa, branch_solutions_dgamma, branch_solutions_dtheta, b0, tree):
    path = get_path(spA,spB,tree)
    db_AB_dkappa = {}
    db_AB_dgamma = {}
    db_AB_dtheta = {}
    for dbranch in path:
        db_AB_dkappa_branch = b0[path[0]]
        db_AB_dgamma_branch = b0[path[0]]
        db_AB_dtheta_branch = b0[path[0]]
        for branch in path:
            if branch == dbranch:
                db_AB_dkappa_branch = branch_solutions_dkappa[branch].dot(db_AB_dkappa_branch)
                db_AB_dgamma_branch = branch_solutions_dgamma[branch].dot(db_AB_dgamma_branch)
                db_AB_dtheta_branch = branch_solutions_dtheta[branch].dot(db_AB_dtheta_branch)
            else:
                db_AB_dkappa_branch = branch_solutions[branch].dot(db_AB_dkappa_branch)
                db_AB_dgamma_branch = branch_solutions[branch].dot(db_AB_dgamma_branch)
                db_AB_dtheta_branch = branch_solutions[branch].dot(db_AB_dtheta_branch)
        db_AB_dkappa[dbranch] = db_AB_dkappa_branch
        db_AB_dgamma[dbranch] = db_AB_dgamma_branch
        db_AB_dtheta[dbranch] = db_AB_dtheta_branch
    return db_AB_dkappa, db_AB_dgamma, db_AB_dtheta
