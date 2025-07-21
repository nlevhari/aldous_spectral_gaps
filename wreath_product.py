import itertools
import numpy as np

###############################################################################
# 1) GROUP CONSTRUCTION HELPERS
###############################################################################

def identity_permutation(n):
    """Return the identity permutation on {0,1,...,n-1} as a tuple."""
    return tuple(range(n))

def permute(perm, i):
    """Given permutation perm (as a tuple), return perm(i)."""
    return perm[i]

def compose_permutations(sigma, tau):
    """
    Return sigma o tau as a tuple, 
    meaning first apply tau, then apply sigma.
    Both sigma and tau are stored as 0-based tuples:
      sigma[i] = sigma(i).
    """
    n = len(sigma)
    # (sigma o tau)(i) = sigma(tau(i))
    return tuple(sigma[tau[i]] for i in range(n))

def invert_permutation(sigma):
    """
    Given sigma as a tuple, return its inverse as a tuple.
    That is inv_sigma[sigma[i]] = i.
    """
    n = len(sigma)
    inv = [None]*n
    for i in range(n):
        inv[sigma[i]] = i
    return tuple(inv)

def group_multiply(g1, g2, m):
    """
    Multiply two wreath product elements g1, g2 in C_m^n rtimes S_n.
    g1 = (sigma1, (a1,...,an))
    g2 = (sigma2, (b1,...,bn))
    We use the formula:
      (sigma1, a) * (sigma2, b) = (sigma1 o sigma2, (a1 + b_{sigma1^-1(1)}, ..., an + b_{sigma1^-1(n)})) mod m
    """
    sigma1, a = g1
    sigma2, b = g2
    n = len(sigma1)
    inv_sigma1 = invert_permutation(sigma1)
    
    # compose permutations
    sigma12 = compose_permutations(sigma1, sigma2)
    
    # add bottom components using the appropriate index shift
    new_a = [None]*n
    for i in range(n):
        # find j = sigma1^-1(i)
        j = inv_sigma1[i]
        new_a[i] = (a[i] + b[j]) % m
    
    return (sigma12, tuple(new_a))

def build_wreath_product(m, n):
    """
    Return:
      group_elements: list of all elements (sigma, (a1,...,an))
      elem_to_idx: map (sigma, (a1,...,an)) -> index
      idx_to_elem: inverse map index -> (sigma, (a1,...,an))
    in C_m wr S_n, enumerating permutations and bottom components.
    """
    # All permutations in S_n
    from math import factorial
    from itertools import permutations, product
    
    perms = list(permutations(range(n)))  # all n! permutations
    # All possible bottom group elements = (a1,...,an) with ai in {0,...,m-1}
    bottom_tuples = list(product(range(m), repeat=n))
    
    group_elements = []
    for sigma in perms:
        for bottom in bottom_tuples:
            group_elements.append((sigma, bottom))
    
    elem_to_idx = {g: i for i, g in enumerate(group_elements)}
    idx_to_elem = {i: g for i, g in enumerate(group_elements)}
    
    return group_elements, elem_to_idx, idx_to_elem



###############################################################################
# 1.  ρ(g):  build once, then cache
###############################################################################
from functools import lru_cache

@lru_cache(maxsize=None)
def rho_matrix(elem, n, m):
    """
    elem  = (sigma, a_tuple)  ∈  C_m^n ⋊ S_n
    Return its  (mn)×(mn)  matrix in the basis
       [(0,0),(0,1),…,(0,n-1), (1,0), …, (m-1,n-1)]
    """
    d   = m * n
    M   = np.zeros((d, d), dtype=float)

    # index helper:  colour-major order
    def idx(x, i):      # x ∈ {0..m-1},  i ∈ {0..n-1}
        return x * n + i

    sigma, a = elem
    inv_sig  = invert_permutation(sigma)

    for x in range(m):
        for i in range(n):
            j      = sigma[i]
            x_new  = (x + a[inv_sig[i]]) % m
            M[idx(x_new, j), idx(x, i)] = 1.0
    return M


###############################################################################
# 2.  ρ⊗ρ(E)  =  Σ_g  α_g · (ρ(g) ⊗ ρ(g))
###############################################################################
def tensor_rep_matrix(E_dict, n, m):
    """
    Build the big  (mn)² × (mn)²  matrix of E in ρ⊗ρ.
    """
    d  = m * n
    big = np.zeros((d*d, d*d), dtype=float)

    for g, coeff in E_dict.items():
        R = rho_matrix(g, n, m)     # pulled from cache or built once
        big += coeff * np.kron(R, R)
    return big

###############################################################################
# 3.  Spectrum helpers
###############################################################################
def eigenvalues_tensor(E_dict, n, m):
    M = tensor_rep_matrix(E_dict, n, m)
    vals = np.linalg.eigvals(M)     # real but not guaranteed symmetric
    return np.sort([round(val,2) for val in vals.real])

def smallest_tensor(E_dict, n, m, tol=1e-12):
    vals = eigenvalues_tensor(E_dict, n, m)
    # skip numerically-zero mode
    return vals[0]


