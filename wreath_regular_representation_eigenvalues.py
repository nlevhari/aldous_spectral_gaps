import numpy as np
from wreath_product import group_multiply


def build_matrix_regular(E_dict, group_elements, elem_to_idx, m):
    """
    Build the (size x size) matrix of E in the regular representation,
    where size = len(group_elements).
    
    E_dict: dictionary describing the element E.
    group_elements: list of all group elements in some canonical order.
    elem_to_idx: dict mapping group_element -> index.
    m: order of the cyclic bottom group.
    """
    size = len(group_elements)
    M = np.zeros((size, size), dtype=float)
    
    # Precompute multiplication with each group element to avoid overhead
    # We'll store multiply_table[h_idx][g_idx] = index of (g*h).
    # But a simpler approach is just to do the nested loop below:
    #
    # For each column h_idx, we look up the group element h:
    for h_idx, h in enumerate(group_elements):
        for g, alpha in E_dict.items():
            u = group_multiply(g.to_tuple(), h, m)  # u = g*h
            u_idx = elem_to_idx[u]
            M[u_idx, h_idx] += alpha
    
    return M


def eigenvalues_regular(E_dict, group_elements, elem_to_idx, m):
    M = build_matrix_regular(E_dict, group_elements, elem_to_idx, m)
    # It's typically real symmetric if c_{i,j} are real 
    # and J_{i,j} is a sum of group elements plus identity, 
    # but let's not assume that. We'll just use eig.
    vals = np.linalg.eigvals(M)
    return sorted([round(val, 2) for val in vals], reverse=True)  # Round for consistency

def second_smallest_eigenvalue(vals, tol=1e-12):
    # Sort the real parts (or magnitudes). Typically they should be real anyway.
    vals_real = np.sort(vals.real)
    # The first is presumably 0 or near 0. Return the next.
    return vals_real[1]
