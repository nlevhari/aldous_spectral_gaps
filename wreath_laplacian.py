from wreath_product import identity_permutation


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
        E_dict[id_elem] = E_dict.get(id_elem, 0.0) + 2*(m**2)*c_val
        
        # Subtract c_{i,j} * J_{i,j}
        Jij = build_J_ij(i, j, n, m)
        for g, coeff_g in Jij.items():
            E_dict[g] = E_dict.get(g, 0.0) - c_val*coeff_g
    
    # Remove any zero-coefficient entries (optional cleanup)
    E_dict = {g:val for g,val in E_dict.items() if abs(val) > 1e-15}
    return E_dict
