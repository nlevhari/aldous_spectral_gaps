import json
import time
import numpy as np
import itertools
import argparse

from Sn_standard_representation_eigenvalues import eigenvalues_standard_sn
from wreath_laplacian import build_element_E
from wreath_product import build_wreath_product
from wreath_regular_representation_eigenvalues import eigenvalues_regular
from wreath_standard_representation_eigenvalues import eigenvalues_Cmxn

def eigenvalue_computation(n, m, randomize_c_ij=False, c_dict_=None, verbose=False):
    def myprint(*args):
        if verbose:
            print(args)
    if c_dict_ is None:    
        c_dict = {}
        for i in range(n):
            for j in range(i+1, n):
                c_dict[(i,j)] = abs(np.random.normal()) if randomize_c_ij else 1.0
    else:
        c_dict = {}
        for i in range(n):
            for j in range(i+1, n):
                c_dict[(i,j)]=c_dict_.pop((i,j), 0.0)
        if c_dict_:
            print("dictionary has leftover bad keys")
            exit(1)
    # 1) Build the element E
    E_dict = build_element_E(c_dict, n, m)
    
    # 2a) Regular representation
    group_elements, elem_to_idx, idx_to_elem = build_wreath_product(m, n)
    vals_reg = eigenvalues_regular(E_dict, group_elements, elem_to_idx, m)
    vals_reg_sorted = np.sort(vals_reg.real)
    # second smallest
    
    myprint("Regular representation: smallest two eigenvalues =",
          vals_reg_sorted[0], vals_reg_sorted[1])
    
    # 2b) Representation on (C_m x [n])
    vals_cmxn = eigenvalues_Cmxn(E_dict, n, m)
    vals_cmxn_sorted = np.sort(vals_cmxn.real)
    
    myprint("C_m x [n] representation: smallest two eigenvalues =",
          vals_cmxn_sorted[0], vals_cmxn_sorted[1])
    
    vals_n = eigenvalues_standard_sn(E_dict, n)
    vals_n_sorted = np.sort(vals_n.real)
    myprint("[n] representation: smallest two eigenvalues =", 
          vals_n_sorted[0], vals_n_sorted[1])
    
    should_be_non_positive = [vals_reg_sorted[0], vals_cmxn_sorted[0], vals_n_sorted[0], vals_reg_sorted[1] - vals_cmxn_sorted[1], vals_cmxn_sorted[1] - vals_n_sorted[1]]
    if (max(should_be_non_positive) > 0.0001):
        print("bugggggg")
        print("[vals_reg[0], vals_cmxn[0], vals_n[0], vals_reg_sorted[1] - vals_cmxn_sorted[1], vals_cmxn_sorted[1] - vals_n_sorted[1]]=\n",should_be_non_positive)
        print("c values:", c_dict)
        exit(1)
    elif vals_cmxn_sorted[1] - vals_reg_sorted[1] > 0.001:
        print(f"diff between Cmx[n] rep and regular: {vals_cmxn_sorted[1] - vals_reg_sorted[1]}")
        print("[n]:", vals_n_sorted[1])
        print("Cmx[n]:", vals_cmxn_sorted[1])
        print("regular:", vals_reg_sorted[1])
        print("c values:", c_dict)
    elif vals_n_sorted[1] - vals_reg_sorted[1] > 0.001:
        print(f"diff between [n] rep and regular: {vals_n_sorted[1] - vals_reg_sorted[1]}")
        print("[n]:", vals_n_sorted[1])
        print("Cmx[n]:", vals_cmxn_sorted[1])
        print("regular:", vals_reg_sorted[1])
        print("c values:", c_dict)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", default=3, help="size of the action set for the symmetric group", type=int)
    p.add_argument("-m", default=2, help="order of the cyclic group", type=int)
    p.add_argument("-r", "--randomize-c-ij", help="flag if one wants to randomize the coefficients with a normal distribution", action="store_true")
    p.add_argument("-s", "--stats", help="flag if one wants to run 100 randomizations of c vectors and check if the spectral gaps are the same", action="store_true")
    p.add_argument("-c",
                    "--c-dict", default=None)
    args = p.parse_args()
    c_dict_ = None
    if args.c_dict is not None:
        with open(args.c_dict, 'r') as f:
            raw = json.load(f)
            c_dict_ = {tuple(map(int, k.strip("()").split(", "))): v for k, v in raw.items()}
    if args.stats:
        for nn in range(3, args.n+1):
            for mm in range(2, args.m+1):
                print(f"running on n = {nn} and m = {mm}")
                start = time.time()
                for i in range(100):
                    eigenvalue_computation(nn, mm, randomize_c_ij=True)
                print("took: ", time.time()-start,"s")
    else:
        eigenvalue_computation(args.n, args.m, randomize_c_ij=args.randomize_c_ij, verbose=True, c_dict_=c_dict_)
