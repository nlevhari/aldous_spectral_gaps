import itertools
import numpy as np
import typing
from wreath_product import WreathProductElement, WreathProductGroupAlgebraElement

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
                yield WreathProductElement(sigma, tuple(a), n, m)

###############################################################################
# 2.  Dictionary form of  β_{ {i,j} }  = 1 - T_{ {i,j} }
###############################################################################

def beta_pair_dict(i,j, m, n, G_elems):
    coeffs = {}       # group_element -> real coefficient
    # start with the identity element, coeff = 1
    id_elem = WreathProductElement(identity_perm(n), tuple([0]*n), n, m)
    coeffs[id_elem] = 2.0

    norm = 1.0 / ( (len(G_elems)**2) * 1.0 )   # (|G|^2)(1!)
    for g in support_T_pair(i,j,m,n,G_elems):
        coeffs[g] = coeffs.get(g,0.0) - norm
    return WreathProductGroupAlgebraElement({g:c for g,c in coeffs.items() if abs(c) > 1e-15})

###############################################################################
# 3.  Assemble the coloured octopus element Ω_G(c) as in (2)
###############################################################################

def coloured_octopus(c_list, m, n, G_elems):
    """
    c_list = [c2,...,cn]  (Python indices 0..n-1; we ignore index 0)
    returns a dict  E[g] = coefficient  describing Ω_G(c).
    """
    E = WreathProductGroupAlgebraElement({})  # empty dict for coefficients
    #
    Ctot = sum(c_list[1:])
    # first term   (Σ c_i)·(Σ c_j β_{1j})
    for j,cj in enumerate(c_list):
        if j == 0:
            continue
        beta = beta_pair_dict(0,j,m,n,G_elems)
        E = E + beta * (Ctot * cj)
    # subtract   Σ_{i<j} c_i c_j β_{ij}
    for i in range(1,n-1):
        for j in range(i+1,n):
            beta = beta_pair_dict(i,j,m,n,G_elems)
            E = E + beta * (-c_list[i]*c_list[j])
    return WreathProductGroupAlgebraElement({g:v for g,v in E.coeffs.items() if abs(v)>1e-15})

def w(i, j, n, m):
    if i<j: return WreathProductGroupAlgebraElement(2, n, m)
    if i==j: return WreathProductGroupAlgebraElement(1, n, m)
    if j<i: exit(3)

def U(i, j, m, n):
    if i==j:
        return beta_pair_dict(0,i,m,n,tuple(range(m)))
    if i<j:
        return beta_pair_dict(0,i,m,n,tuple(range(m))) + beta_pair_dict(0,j,m,n,tuple(range(m))) - beta_pair_dict(i,j,m,n,tuple(range(m)))
    if j<i:
        exit(4)

def Jordan(op1:WreathProductGroupAlgebraElement, op2:WreathProductGroupAlgebraElement):
    """
    Returns the Jordan product of two operators op1 and op2 in the wreath product algebra.
    """
    # op1 and op2 are dictionaries mapping group elements to coefficients
    return ((op1*op2 + op2*op1) / 2)

def Lambda(e1, f1, e2, f2, m, n):
    """
    Returns the Lambda operator for the wreath product algebra.
    e1, f1, e2, f2 are indices.
    """
    return w(e1, f1, n, m) * w(e2, f2, n, m) - Jordan((U(e1, f1, m, n) - w(e1, f1, n, m)), (U(e2, f2, m, n) - w(e2, f2, n, m)))

def Gamma_2222(m, n):
    """
    Returns the Gamma operator for the wreath product algebra.
    This is a specific case of the Lambda operator where all indices are 2.
    """
    return Lambda(1,1,1,1,m,n)

def Gamma_2223(m, n):
    """
    Returns the Gamma operator for the wreath product algebra.
    This is a specific case of the Lambda operator where indices are 2,2,2,3.
    """
    return Lambda(1,1,1,2,m,n) * 2 # (second one is 1,2,1,1) 

def Gamma_2233(m, n):
    """
    Returns the Gamma operator for the wreath product algebra.
    This is a specific case of the Lambda operator where indices are 2,2,3,3.
    """
    return (Lambda(1,1,2,2,m,n) * 2 # second one is 2,2,1,1
            + Lambda(1,2,1,2,m,n))

def Gamma_2234(m, n):
    """
    Returns the Gamma operator for the wreath product algebra.
    This is a specific case of the Lambda operator where indices are 2,2,3,4.
    """
    return (Lambda(1,1,2,3,m,n) * 2 # second one is 2,3,1,1
            + Lambda(1,2,1,3,m,n) * 2) # second one is 1,3,1,2

def Gamma_2345(m, n):
    """
    Returns the Gamma operator for the wreath product algebra.
    This is a specific case of the Lambda operator where indices are 2,3,4,5.
    """
    return (Lambda(1,2,3,4,m,n) * 2 # second one is 3,4,1,2
            + Lambda(1,3,2,4,m,n) * 2 # second one is 2,4,1,3
            + Lambda(1,4,2,3,m,n) * 2) # second one is 2,3,1,4

Gammas = [(Gamma_2222, "Gamma_2222"), (Gamma_2223, "Gamma_2223"), (Gamma_2233, "Gamma_2233"), (Gamma_2234, "Gamma_2234"), (Gamma_2345, "Gamma_2345")]