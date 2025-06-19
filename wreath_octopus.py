import itertools
import numpy as np

###############################################################################
# 0.  Utilities for the wreath product  G wr S_n  (same helpers as before)
###############################################################################

def identity_perm(n):           return tuple(range(n))
def invert_perm(p):             return tuple(np.argsort(p))
def compose(p,q):               return tuple(p[i] for i in q)

def mul(g1, g2, m, n):
    """
    Multiply g1 = (sigma, a) and g2 = (tau, b) in G^n ⋊ S_n,
    where bottom colours are taken mod m.
    """
    sigma, a = g1         # unpack here
    tau,   b = g2

    inv_sigma = invert_perm(sigma)
    new_sigma = compose(sigma, tau)

    new_a = [(a[i] + b[inv_sigma[i]]) % m for i in range(n)]
    return (new_sigma, tuple(new_a))


###############################################################################
# 1.  All (g_i,g_j,π) elements that make up T_{ {i,j} }
###############################################################################

def support_T_pair(i,j, m, n, G_elems):
    """Iterator over the |G|^2 · 2 elements x_{ {i,j}; g_i,g_j; π }"""
    idp  = identity_perm(n)
    swap = list(idp); swap[i],swap[j]=swap[j],swap[i]; swap=tuple(swap)
    for gi in G_elems:
        for gj in G_elems:
            for sigma in (idp, swap):
                # colour vector
                a = [0]*n
                a[i] = gi   # store colours as integers 0..|G|-1 for speed
                a[j] = gj
                yield (sigma, tuple(a))

###############################################################################
# 2.  Dictionary form of  β_{ {i,j} }  = 1 - T_{ {i,j} }
###############################################################################

def beta_pair_dict(i,j, m, n, G_elems):
    coeffs = {}       # group_element -> real coefficient
    # start with the identity element, coeff = 1
    id_elem = (identity_perm(n), tuple([0]*n))
    coeffs[id_elem] = 2.0

    norm = 1.0 / ( (len(G_elems)**2) * 1.0 )   # (|G|^2)(1!)
    for g in support_T_pair(i,j,m,n,G_elems):
        coeffs[g] = coeffs.get(g,0.0) - norm
    return {g:c for g,c in coeffs.items() if abs(c) > 1e-15}

###############################################################################
# 3.  Assemble the coloured octopus element Ω_G(c) as in (2)
###############################################################################

def coloured_octopus(c_list, m, n, G_elems):
    """
    c_list = [c2,...,cn]  (Python indices 0..n-1; we ignore index 0)
    returns a dict  E[g] = coefficient  describing Ω_G(c).
    """
    E = {}
    # helper to add dictionaries
    def add_into(target, source, scale):
        for g,v in source.items():
            target[g] = target.get(g,0.0) + scale*v
    #
    Ctot = sum(c_list[1:])
    # first term   (Σ c_i)·(Σ c_j β_{1j})
    for j,cj in enumerate(c_list):
        if j == 0:
            continue
        beta = beta_pair_dict(0,j,m,n,G_elems)
        add_into(E, beta, Ctot*cj)
    # subtract   Σ_{i<j} c_i c_j β_{ij}
    for i in range(1,n-1):
        for j in range(i+1,n):
            beta = beta_pair_dict(i,j,m,n,G_elems)
            add_into(E, beta, -c_list[i]*c_list[j])
    return {g:v for g,v in E.items() if abs(v)>1e-15}
