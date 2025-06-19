import numpy as np
from wreath_product import identity_permutation

def create_c_dict_ij(n, randomize_c=False):
    c_dict_E = {}
    for i in range(n):
        for j in range(i+1, n):
            c_dict_E[(i,j)] = abs(np.random.normal()) if randomize_c else 1.0
    return c_dict_E

def fill_and_validate_c_dict_ij(n, c_dict_):
    c_dict_E = {}
    for i in range(n):
        for j in range(i+1, n):
            c_dict_E[(i,j)]=c_dict_.pop((i,j), 0.0)
    if c_dict_:
        print("dictionary has leftover bad keys")
        exit(1)
    return c_dict_E

def build_J_ij(i, j, n, m):
    """
    Build the dictionary for J_{i,j}: 
      sum of all group elements that
        - are identity on all rows except possibly i, j,
        - in the top group, either do nothing or swap i<->j,
        - in the bottom group, any (a_i, a_j) in range(m)^2 for those indices.
    We'll use 0-based indexing for rows i,j.
    Return: A dict { group_element : 1.0, ... } 
    """
    # The identity permutation on {0,...,n-1}:
    idperm = identity_permutation(n)
    
    # We define two relevant permutations in that "2x2" block: 
    #   - identity
    #   - transposition that swaps i <-> j
    def swap_ij(perm, i, j):
        perm = list(perm)
        perm[i], perm[j] = perm[j], perm[i]
        return tuple(perm)
    
    sigma_id = idperm
    sigma_swap = swap_ij(idperm, i, j)
    
    J_ij = {}
    
    # For bottom components: choose anything for i, j, 0 for others
    import itertools
    for top_perm in (sigma_id, sigma_swap):
        for a_i, a_j in itertools.product(range(m), repeat=2):
            # Build the bottom tuple
            bottom = [0]*n
            bottom[i] = a_i
            bottom[j] = a_j
            bottom = tuple(bottom)
            g = (top_perm, bottom)
            J_ij[g] = J_ij.get(g, 0.0) + 1.0
    
    return J_ij

def build_element_E(c_dict, n, m):
    """
    c_dict is a dict of the form c_dict[(i,j)] = c_{i,j}, 
      for 0 <= i < j < n, real coefficients.
    We build E = sum_{i<j} c_{i,j} ( 2*(m**2) id - J_{i,j} ).
    
    Return: a dict E_dict with E_dict[g] = coefficient in front of g.
    """
    E_dict = {}
    
    # identity element
    idperm = identity_permutation(n)
    id_bottom = tuple([0]*n)
    id_elem = (idperm, id_bottom)
    
    for (i, j), c_val in c_dict.items():
        # Add c_{i,j} * (2*m^2 id)
        E_dict[id_elem] = E_dict.get(id_elem, 0.0) + 2*c_val
        
        # Subtract c_{i,j} * J_{i,j}
        Jij = build_J_ij(i, j, n, m)
        for g, coeff_g in Jij.items():
            E_dict[g] = E_dict.get(g, 0.0) - c_val*coeff_g / (m**2)
    
    # Remove any zero-coefficient entries (optional cleanup)
    E_dict = {g:val for g,val in E_dict.items() if abs(val) > 1e-15}
    return E_dict

def swap_ij(perm, i, j):
    perm = list(perm)
    perm[i], perm[j] = perm[j], perm[i]
    return tuple(perm)


def build_element_F(x_dict, y_dict, alpha_dict, n, m):
    """
    c_dict is a dict of the form c_dict[(i,j)] = x_{i,j}, alpha_dict[g] = alpha_{zeta^g}, y_dict[(-1, w)] = y_w
      for 0 <= i < j < n, 0<=g<m, 0<=w<n real coefficients.
    We build F = sum_{i<j} x_{i,j} (i, j) + sum_w sum_g alpha_g * g^(w).
    g^(w) is g in the w'th index.
    
    Return: a dict F_dict with F_dict[g] = coefficient in front of g.
    """
    F_dict = {}
    
    # identity element
    idperm = identity_permutation(n)
    id_bottom = tuple([0]*n)
    id_elem = (idperm, id_bottom)
    
    for (i,j), x_ij in x_dict.items():
        # Add c_{i,j} * (2id - id)
        F_dict[id_elem] = F_dict.get(id_elem, 0.0) + x_ij
        
        # Subtract c_ij * (ij)
        
        sigma_swap = swap_ij(idperm, i, j)
        g = (sigma_swap, tuple([0]*n))
        F_dict[g] = F_dict.get(g, 0.0) - x_ij
    
    for w, y_w in y_dict.items():
        for g, alpha_g in alpha_dict.items():
            F_dict[id_elem] = F_dict.get(id_elem, 0.0) + y_w * alpha_g

            bottom = [0]*n
            bottom[w] = g
            h = (idperm, tuple(bottom))
            F_dict[h] = F_dict.get(h, 0.0) - y_w * alpha_g
    
    # Remove any zero-coefficient entries (optional cleanup)
    F_dict = {g:val for g,val in F_dict.items() if abs(val) > 1e-15}
    return F_dict

def get_E_element(n, m, c_dict_=None, randomize_c=False):
    c_dict_E = None
    if c_dict_ is None:    
        c_dict_E = create_c_dict_ij(n ,randomize_c)
    else:
        c_dict_E = fill_and_validate_c_dict_ij(n, c_dict_)
    # 1) Build the element E
    E_dict = build_element_E(c_dict_E, n, m)
    return E_dict, c_dict_E

def create_alpha_dict(m, randomize_c=False):
    alpha_dict = {}
    for i in range(m//2 + 1):
        alpha_dict[i] = abs(np.random.normal()) if randomize_c else 1.0
        if i!= 0 and m-i!=i:
            alpha_dict[m-i] = alpha_dict[i]
    return alpha_dict

def create_y_dict(n, randomize_c=False):
    y_dict = {}
    for i in range(n):
        y_dict[i] = abs(np.random.normal()) if randomize_c else 1.0
    return y_dict

def get_F_element(n, m, randomize_c=False):
    x_dict_F = create_c_dict_ij(n ,randomize_c)
    alpha_dict_F = create_alpha_dict(m, randomize_c)
    y_dict_F = create_y_dict(n, randomize_c)
    # 1) Build the element E
    F_dict = build_element_F(x_dict_F, y_dict_F, alpha_dict_F, n, m)
    return F_dict, x_dict_F, alpha_dict_F, y_dict_F