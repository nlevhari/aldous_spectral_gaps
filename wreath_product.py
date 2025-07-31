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

def group_multiply(g1, g2, n, m):
    """
    Multiply two wreath product elements g1, g2 in C_m^n rtimes S_n.
    g1 = (sigma1, (a1,...,an))
    g2 = (sigma2, (b1,...,bn))
    We use the formula:
      (sigma1, a) * (sigma2, b) = (sigma1 o sigma2, (a1 + b_{sigma1^-1(1)}, ..., an + b_{sigma1^-1(n)})) mod m
    """
    sigma1, a = g1
    sigma2, b = g2
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

def build_wreath_product(n, m):
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
            group_elements.append(WreathProductElement(sigma, bottom, n, m))
    
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

    sigma, a = elem.to_tuple()
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
def tensor_rep_matrix(E_element, n, m):
    """
    Build the big  (mn)² × (mn)²  matrix of E in ρ⊗ρ.
    """
    d  = m * n
    big = np.zeros((d*d, d*d), dtype=float)

    for g, coeff in E_element.coeffs.items():
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


class WreathProductElement:
    def __init__(self, sigma, a_tuple, n, m):
        """
        Initialize a wreath product element.
        sigma: permutation tuple in S_n
        a_tuple: bottom component tuple in C_m^n
        """
        self.sigma = sigma
        self.a_tuple = tuple(a_tuple)
        self.n = n
        self.m = m

    def __repr__(self):
        return f"WreathProductElement(sigma={self.sigma}, a_tuple={self.a_tuple})"

    def __eq__(self, other):
        return (self.sigma == other.sigma) and (self.a_tuple == other.a_tuple)

    def __hash__(self):
        return hash((self.sigma, self.a_tuple))
    
    def to_tuple(self):
        """
        Convert to tuple representation for hashing or storage.
        """
        return (self.sigma, self.a_tuple)

    @staticmethod
    def from_tuple(tup, n, m):
        """
        Convert from tuple representation to WreathProductElement.
        """
        return WreathProductElement(tup[0], tup[1], n, m)

    def __mul__(self, other):
        """
        Multiply two wreath product elements.
        """
        if isinstance(other, WreathProductElement):
            return WreathProductElement.from_tuple(group_multiply(self.to_tuple(), other.to_tuple(), self.n, self.m), self.n, self.m)
        else:
            raise TypeError("Can only multiply with another WreathProductElement")

class WreathProductGroupAlgebraElement:
    def __init__(self, coeffs, n=0, m=None):
        """
        Initialize a group algebra element.
        coeffs: dictionary mapping WreathProductElement to coefficient or scalar (in that case, n should be input).
        n: size of the group symmetrized, used to create the identity element if coeffs is a scalar. 
        """
        if n!=0:
            if m is None:
                raise ValueError("If n is specified, m must also be specified.")
            self.coeffs = {WreathProductElement.from_tuple((identity_permutation(n), [0]*n), n, m): coeffs}
        else:
            self.coeffs = coeffs

    def __repr__(self):
        return f"WreathProductGroupAlgebraElement(coeffs={self.coeffs})"
    
    def __eq__(self, other):
        visited = set()
        for k in self.coeffs:
            if not np.isclose(self.coeffs.get(k, 0), other.coeffs.get(k, 0)):
                return False
            visited.add(k)
        for k in other.coeffs:
            if k not in visited and not np.isclose(other.coeffs.get(k, 0), 0):
                return False
        return True
    
    def __add__(self, other):
        """
        Add two group algebra elements.
        """
        new_coeffs = self.coeffs.copy()
        for k, v in other.coeffs.items():
            new_coeffs[k] = new_coeffs.get(k, 0) + v
        return WreathProductGroupAlgebraElement(new_coeffs)
    
    def __sub__(self, other):
        """
        Subtract two group algebra elements.
        """
        new_coeffs = self.coeffs.copy()
        for k, v in other.coeffs.items():
            new_coeffs[k] = new_coeffs.get(k, 0) - v
        return WreathProductGroupAlgebraElement(new_coeffs)
    
    def __mul__(self, other):
        result = {}
        if type(other) is not WreathProductGroupAlgebraElement:
            # Scalar multiplication
            for k, v in self.coeffs.items():
                result[k] = v * other
            return WreathProductGroupAlgebraElement(result)
        for k1, v1 in self.coeffs.items():
            for k2, v2 in other.coeffs.items():
                product_elem = k1*k2
                result[product_elem] = result.get(product_elem, 0) + v2 * v1
        return WreathProductGroupAlgebraElement(result)
    
    def __truediv__(self, scalar):
        """
        Divide all coefficients by a scalar.
        """
        if scalar <= 0.001:
            raise ZeroDivisionError("Cannot divide by zero or a very small number.")
        new_coeffs = {k: v / scalar for k, v in self.coeffs.items()}
        return WreathProductGroupAlgebraElement(new_coeffs)
    
    def get_dict(self):
        """
        Return the dictionary of coefficients.
        """
        return self.coeffs