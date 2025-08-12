# sn_group_algebra.py
import numpy as np
from itertools import permutations, combinations
from wreath_product import identity_permutation, compose_permutations

class SnGroupAlgebraElement:
    def __init__(self, coeffs, n: int = 0):
        """
        coeffs: dict[perm(tuple) -> float] or a scalar if n>0 (places it on id).
        If n>0 and coeffs is scalar, builds coeffs[id] = coeffs.
        """
        if n > 0 and not isinstance(coeffs, dict):
            idp = identity_permutation(n)
            self.coeffs = {idp: float(coeffs)}
        elif isinstance(coeffs, dict):
            self.coeffs = {tuple(k): float(v) for k, v in coeffs.items() if abs(v) > 1e-15}
        else:
            raise ValueError("Provide dict of coeffs or scalar with n>0")

    def __repr__(self):
        return f"SnGroupAlgebraElement(coeffs={self.coeffs})"

    def get_dict(self):
        return self.coeffs

    def __eq__(self, other):
        if not isinstance(other, SnGroupAlgebraElement):
            return False
        keys = set(self.coeffs) | set(other.coeffs)
        for k in keys:
            if not np.isclose(self.coeffs.get(k, 0.0), other.coeffs.get(k, 0.0)):
                return False
        return True

    def __add__(self, other):
        out = self.coeffs.copy()
        for k, v in other.coeffs.items():
            out[k] = out.get(k, 0.0) + v
        return SnGroupAlgebraElement(out)

    def __sub__(self, other):
        out = self.coeffs.copy()
        for k, v in other.coeffs.items():
            out[k] = out.get(k, 0.0) - v
        return SnGroupAlgebraElement(out)

    def __mul__(self, other):
        # scalar
        if not isinstance(other, SnGroupAlgebraElement):
            return SnGroupAlgebraElement({k: v * other for k, v in self.coeffs.items()})
        # convolution in C[S_n]: (Σ v1·p1)(Σ v2·p2) = Σ (v1 v2)·(p1∘p2)
        out = {}
        for p1, v1 in self.coeffs.items():
            for p2, v2 in other.coeffs.items():
                p = compose_permutations(p1, p2)
                out[p] = out.get(p, 0.0) + v1 * v2
        return SnGroupAlgebraElement(out)

    def __truediv__(self, scalar: float):
        if abs(scalar) < 1e-12:
            raise ZeroDivisionError("Division by zero.")
        return SnGroupAlgebraElement({k: v / scalar for k, v in self.coeffs.items()})

def build_J_ijk_sn(i: int, j: int, k: int, n: int) -> SnGroupAlgebraElement:
    """
    J_{i,j,k} = sum of all perms fixing everything outside {i,j,k}.
    6 terms (embedded S_3).
    """
    J = {}
    triple = (i, j, k)
    for img in permutations(triple):  # mapping i->img[0], j->img[1], k->img[2]
        sigma = list(range(n))
        sigma[i], sigma[j], sigma[k] = img
        t = tuple(sigma)
        J[t] = J.get(t, 0.0) + 1.0
    return SnGroupAlgebraElement(J)

def create_c_dict_ijk(n: int, randomize_c: bool = False):
    rng = np.random.default_rng()
    c = {}
    for (i, j, k) in combinations(range(n), 3):
        c[(i, j, k)] = float(abs(rng.normal())) if randomize_c else 1.0
    return c

def build_E_sn_triplets(c_dict_ijk, n: int) -> SnGroupAlgebraElement:
    """
    E_sn = Σ_{i<j<k} c_{ijk} · (3·id − 1/2·J_{ijk})
    """
    id_elem = SnGroupAlgebraElement(1.0, n=n)  # 1·id
    E = SnGroupAlgebraElement({}, n=0)
    for (i, j, k), c in c_dict_ijk.items():
        E = E + (id_elem * (3.0 * c))
        J = build_J_ijk_sn(i, j, k, n)
        E = E + (J * (-0.5 * c))
    # prune zeros
    return SnGroupAlgebraElement({g: v for g, v in E.get_dict().items() if abs(v) > 1e-15})