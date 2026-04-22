import numpy as np
from scipy import stats
from sasquatch.tools import *

def simulate_neutral_genome_evolution(karyotype, T, kappa, gamma, theta, return_trajectory = False):
    genes = np.sum(karyotype)
    G = [genome_from_karyotype(karyotype)]
    N = len(G[0])
    K = np.sum([len(g)-1 for g in G[0]])
    q = 1 - np.exp(-theta)
    rng = np.random.default_rng()
        
    t = 0
    t_list = [t]
    while t < T:
        G0 = G[-1]
        dt_kappa = stats.expon.rvs(scale=1/(kappa*K))
        dt_gamma = stats.expon.rvs(scale=1/(gamma*K))
        if dt_kappa < dt_gamma:
            cut = 1
            dt = dt_kappa
        else:
            cut = 2
            dt = dt_gamma
            
        p_chrom = np.array([len(g)-1 for g in G0])
        p_chrom = p_chrom / p_chrom.sum()
        r_chrom = rng.random()
        chrom_choice = np.digitize(r_chrom,np.cumsum(p_chrom))
        cut_chrom = G0[chrom_choice]
        if cut == 1:
            cut_choice = rng.choice(range(1,len(cut_chrom)))
            if len(cut_chrom[:cut_choice]) < len(cut_chrom[cut_choice:]):
                chrom_piece_B = cut_chrom[:cut_choice]
                chrom_piece_A = cut_chrom[cut_choice:]
            else:
                chrom_piece_A = cut_chrom[:cut_choice]
                chrom_piece_B = cut_chrom[cut_choice:]
            chrom_piece_to_fuse = 1
            chrom_side_to_fuse_to = rng.choice([0,1])
            
        elif cut == 2:
            cut_choice = np.random.choice(range(0,len(cut_chrom)+1))
            chrom_piece_A = cut_chrom[:cut_choice]
            chrom_piece_BC = cut_chrom[cut_choice:]
            p_cut_size = stats.geom.pmf(np.arange(1,len(chrom_piece_BC)+1),q)
            p_cut_size = p_cut_size / p_cut_size.sum()
            r_cut_size = rng.random()
            cut_size = np.digitize(r_cut_size,np.cumsum(p_cut_size)) + 1
            chrom_piece_B = chrom_piece_BC[:cut_size]
            chrom_piece_C = chrom_piece_BC[cut_size:]            
            chrom_piece_A = chrom_piece_A + chrom_piece_C
            chrom_piece_B = chrom_piece_B
            chrom_piece_to_fuse = 1
     
        r_chrom_fuse = rng.random()
        if cut == 1:
            chrom_choice_fusion = np.digitize(r_chrom_fuse,np.arange(1,N+1)/N)
        elif cut == 2:
            p_chrom = np.array([len(g)-1 for g in G0])
            p_chrom[chrom_choice] = len(chrom_piece_A)
            p_chrom = p_chrom / p_chrom.sum()
            chrom_choice_fusion = np.digitize(r_chrom_fuse,np.cumsum(p_chrom))
        
        G1 = []
        for chrom in range(N):
            if (chrom == chrom_choice) and (chrom != chrom_choice_fusion):
                G1.append(chrom_piece_A)
            elif (chrom != chrom_choice) and (chrom != chrom_choice_fusion):
                G1.append(G0[chrom])
            elif (chrom == chrom_choice) and (chrom == chrom_choice_fusion):
                if cut == 1:
                    if chrom_side_to_fuse_to == 0:
                        G1.append(chrom_piece_B + chrom_piece_A)
                    else:
                        G1.append(chrom_piece_A + chrom_piece_B)
                elif cut == 2:
                    insertion_choice = np.random.choice(range(0,len(chrom_piece_A)+1))
                    G1.append(chrom_piece_A[:insertion_choice] + chrom_piece_B + chrom_piece_A[insertion_choice:])
            else:
                if cut == 1:
                    if chrom_side_to_fuse_to == 0:
                        G1.append(chrom_piece_B + G0[chrom])
                    else:
                        G1.append(G0[chrom] + chrom_piece_B)
                elif cut == 2:
                    insertion_choice = np.random.choice(range(0,len(G0[chrom])+1))
                    G1.append(G0[chrom][:insertion_choice] + chrom_piece_B + G0[chrom][insertion_choice:])
        G.append(G1)
        t += dt
        t_list.append(t)
        
        genes_new = np.sum([len(g) for g in G1])
        if genes_new != genes:
            raise ValueError('Lost some genes!')
        
    if t > T:
        t_list.append(t)
        G.append(G[-1])

    if return_trajectory == False:
        return t, G[-1]
    else:
        return t_list, G
