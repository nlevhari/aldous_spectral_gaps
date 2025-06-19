import json
import time
import numpy as np
import itertools
import argparse

from Sn_standard_representation_eigenvalues import eigenvalues_standard_sn
from wreath_laplacian import get_E_element, get_F_element
from wreath_octopus import coloured_octopus
from wreath_product import build_wreath_product, smallest_tensor
from wreath_regular_representation_eigenvalues import eigenvalues_regular
from wreath_standard_representation_eigenvalues import eigenvalues_Cmxn

def run_single_element_check(n, m, E_dict, c_dict_E, verbose=False):
    def myprint(*args):
        if verbose:
            print(args)
    
    # 2a) Regular representation
    group_elements, elem_to_idx, idx_to_elem = build_wreath_product(m, n)
    vals_reg = eigenvalues_regular(E_dict, group_elements, elem_to_idx, m)
    vals_reg_sorted = np.round(np.sort(vals_reg.real), 2)
    # second smallest
    
    myprint("Regular representation: smallest two eigenvalues =",
          vals_reg_sorted[0], vals_reg_sorted[1])
    
    # 2b) Representation on (C_m x [n])
    vals_cmxn = eigenvalues_Cmxn(E_dict, n, m)
    vals_cmxn_sorted = np.round(np.sort(vals_cmxn.real), 2)
    
    myprint("C_m x [n] representation: smallest two eigenvalues =",
          vals_cmxn_sorted[0], vals_cmxn_sorted[1])
    
    #2c) Representation on [n]
    vals_n = eigenvalues_standard_sn(E_dict, n)
    vals_n_sorted = np.round(np.sort(vals_n.real), 2)
    myprint("[n] representation: smallest two eigenvalues =", 
          vals_n_sorted[0], vals_n_sorted[1])
    # print("[n]:", vals_n_sorted)
    # print("Cmx[n]:", vals_cmxn_sorted)
    # print("regular:", vals_reg_sorted)
    should_be_non_positive = [vals_reg_sorted[0], vals_cmxn_sorted[0], vals_n_sorted[0], vals_reg_sorted[1] - vals_cmxn_sorted[1], vals_cmxn_sorted[1] - vals_n_sorted[1]]
    if (max(should_be_non_positive) > 0.0001):
        print("bugggggg")
        print("[vals_reg[0], vals_cmxn[0], vals_n[0], vals_reg_sorted[1] - vals_cmxn_sorted[1], vals_cmxn_sorted[1] - vals_n_sorted[1]]=\n",should_be_non_positive)
        print("c values:", c_dict_E)
        exit(1)
    elif vals_cmxn_sorted[1] - vals_reg_sorted[1] > 0.001:
        print(f"diff between Cmx[n] rep and regular: {vals_cmxn_sorted[1] - vals_reg_sorted[1]}")
        print("[n]:", vals_n_sorted[1])
        print("Cmx[n]:", vals_cmxn_sorted[1])
        print("regular:", vals_reg_sorted[1])
        print("c values:", c_dict_E)
    elif vals_n_sorted[1] - vals_reg_sorted[1] > 0.001:
        print(f"diff between [n] rep and regular: {vals_n_sorted[1] - vals_reg_sorted[1]}")
        print("[n]:", vals_n_sorted[1])
        print("Cmx[n]:", vals_cmxn_sorted[1])
        print("regular:", vals_reg_sorted[1])
        print("c values:", c_dict_E)

def eigenvalue_computation(n, m, randomize_c=False, c_dict_=None, verbose=False, check_E_element=True, check_F_element=False):
    
    if check_E_element:
        E_dict, c_dict_E = get_E_element(n, m, randomize_c=randomize_c, c_dict_=c_dict_)
        run_single_element_check(n, m, E_dict, c_dict_E)
    elif check_F_element:
        F_dict, _, _, _ = get_F_element(n, m, randomize_c=randomize_c)
        run_single_element_check(n, m, F_dict, None, verbose=True)
    else:
        print("Did not ask to check anything. Exiting")
        exit(0)

def check_octopus(n, m):
    smallest_tensors = []
    for nn in range(3, n+1):
        for mm in range(2, m+1):
            print(f"running on n = {nn} and m = {mm}")
            start = time.time()
            for k in range(10):
                print(f"iteration {k}:")
                import random

                mean = 0.0
                std_dev = 1.0
                coefficients = [abs(random.gauss(mean, std_dev)) for _ in range(nn)]
                random.shuffle(coefficients)
                print(coefficients)
                E_dict = coloured_octopus(coefficients, mm, nn, list(range(mm)))

                print("smallest eigenvalue in ρ⊗ρ:",
                        smallest_tensor(E_dict, nn, mm))
                print(f"coefficients: {coefficients}")
                print("took: ", time.time()-start,"s")
                smallest_tensors += [smallest_tensor(E_dict, nn, mm)]
    print(f"smallest of all is {min(smallest_tensors)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", default=3, help="size of the action set for the symmetric group", type=int)
    p.add_argument("-m", default=2, help="order of the cyclic group", type=int)
    p.add_argument("-r", "--randomize-c", help="flag if one wants to randomize the coefficients with a normal distribution", action="store_true")
    p.add_argument("-s", "--stats", help="flag if one wants to run 100 randomizations of c vectors and check if the spectral gaps are the same", action="store_true")
    p.add_argument("-c",
                    "--c-dict", default=None)
    p.add_argument("-co", "--check-octopus", action="store_true")
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
                    eigenvalue_computation(nn, mm, randomize_c=True)
                print("took: ", time.time()-start,"s")
    if args.check_octopus:
        check_octopus(args.n, args.m)

    else:
        eigenvalue_computation(args.n, args.m, randomize_c=args.randomize_c, verbose=True, c_dict_=c_dict_)
