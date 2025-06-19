import numpy as np


def project_to_Sn(E_dict):
    """
    Given E in the group algebra of C_m wr S_n stored as 
      E_dict[(sigma, a)] = alpha,
    return E_sn_dict, a dictionary in the group algebra of S_n:
      E_sn_dict[sigma] = sum_{a} alpha_{(sigma,a)}.
    """
    from collections import defaultdict
    E_sn_dict = defaultdict(float)
    for (sigma, a), alpha in E_dict.items():
        E_sn_dict[sigma] += alpha
    return dict(E_sn_dict)

def build_perm_rep_matrix(E_sn_dict, n):
    """
    Build the n x n matrix for the element E in the permutation 
    representation of S_n on R^n.
    
    E_sn_dict is a dict: E_sn_dict[sigma] = coefficient
    sigma is stored as a tuple of length n with 0-based indexing:
       sigma[i] = sigma(i).
    We'll build an n x n matrix M.
    Then M[:, i] = sum_{sigma} coefficient(sigma)* e_{sigma(i)}.
    """
    M = np.zeros((n, n), dtype=float)
    
    # We assume permutations are 0-based: sigma: {0,...,n-1}->{0,...,n-1}.
    for i in range(n):
        # Column i
        col_vec = np.zeros(n, dtype=float)
        for sigma, alpha in E_sn_dict.items():
            # row index is sigma(i)
            row = sigma[i]
            col_vec[row] += alpha
        # place col_vec as column i
        M[:, i] = col_vec
    return M

def eigenvalues_standard_sn(E_dict, n):
    """
    Given E in the group algebra of C_m wr S_n (as a dict E_dict), 
    compute the eigenvalues of E in the [n] representation of S_n
    that comes from simply ignoring the bottom C_m^n part and 
    letting the top S_n act by permuting [n].
    
    Steps:
      1) Project E to S_n by summing over all bottom elements.
      2) Build the n x n permutation-representation matrix M.
      3) Calculate the eigenvalues.
    """
    # 1) Project to S_n
    E_sn_dict = project_to_Sn(E_dict)
    
    # 2) Build n x n permutation matrix
    M = build_perm_rep_matrix(E_sn_dict, n)
    
    vals, vecs = np.linalg.eig(M)
    vals = [round(v,2) for v in vals]
    
    return vals

