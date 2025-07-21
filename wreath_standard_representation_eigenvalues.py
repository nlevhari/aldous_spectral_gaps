import numpy as np
from wreath_product import invert_permutation


def wreath_action_on_Cmxn(g, x, i, m):
    """
    A possible standard action: g = (sigma,(a_0,...,a_{n-1}))
    acts on (x, i) in (C_m x [n]) as
       g.(x,i) = ( x + a_{sigma^-1(i)},   sigma(i) )
    (using 0-based indexing).
    Return the resulting pair in C_m x [n].
    """
    sigma, a_tuple = g
    n = len(sigma)
    inv_sigma = invert_permutation(sigma)
    j = inv_sigma[i]   # index so that sigma(j) = i
    # new bottom element:
    x_new = (x + a_tuple[j]) % m
    i_new = sigma[i]
    return (x_new, i_new)

def build_rep_Cmxn_matrix(E_dict, n, m):
    """
    Build the matrix of E in the representation of size m*n
    that comes from the natural action on C_m x [n].
    
    We fix an ordering of (x,i) in the set {0..m-1} x {0..n-1}.
    Then for each column representing basis vector (x,i), we 
    compute E.(x,i) = sum_{g in supp(E)} coeff(g)*[g.(x,i)].
    
    Return M as a 2D numpy array of size (m*n, m*n).
    We also return a dictionary basis_to_idx mapping (x,i)->index 
    so that we know how we are ordering the basis.
    """
    # We'll fix a simple ordering: 
    #   (x=0,i=0), (x=0,i=1),..., (x=0,i=n-1),
    #   (x=1,i=0), (x=1,i=1),..., (x=1,i=n-1),
    #   ...
    basis = []
    for x in range(m):
        for i in range(n):
            basis.append((x,i))
    basis_to_idx = {v: idx for idx,v in enumerate(basis)}
    
    size = m*n
    M = np.zeros((size, size), dtype=float)
    
    # For each column vector (x,i), see where E sends it
    for col_idx, (x,i) in enumerate(basis):
        # E.(x,i) = sum_{g} E_dict[g] * g.(x,i)
        val_dict = {}  # map (x', i') -> total coefficient
        for g, alpha in E_dict.items():
            (x_new, i_new) = wreath_action_on_Cmxn(g, x, i, m)
            val_dict[(x_new, i_new)] = val_dict.get((x_new, i_new), 0.0) + alpha
        
        # Now fill matrix column
        for (x_new, i_new), coeff_val in val_dict.items():
            row_idx = basis_to_idx[(x_new, i_new)]
            M[row_idx, col_idx] += coeff_val
    
    return M, basis_to_idx

def eigenvalues_Cmxn(E_dict, n, m):
    M, basis_to_idx = build_rep_Cmxn_matrix(E_dict, n, m)
    vals, vecs = np.linalg.eig(M)
    vals = sorted([round(val, 2) for val in vals], reverse=True)
    return vals
