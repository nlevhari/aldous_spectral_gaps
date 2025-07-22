import json
import time
import numpy as np
import itertools
import argparse
from pprint import pprint

from Sn_standard_representation_eigenvalues import eigenvalues_standard_sn
from wreath_laplacian import get_E_element, get_F_element
from wreath_octopus import Gammas, coloured_octopus
from wreath_product import build_wreath_product, eigenvalues_tensor, smallest_tensor
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
    group_elements, elem_to_idx, _ = build_wreath_product(m, n)
    for gamma in Gammas:
        gamma_dict = gamma[0](m,n).get_dict()
        e_vals_regular = eigenvalues_regular(gamma_dict, group_elements, elem_to_idx, m)
        print(f"eigenvals 0,1,-1 for {gamma[1]}:",e_vals_regular[0], e_vals_regular[1], e_vals_regular[-1])
        # if gamma[1] == "Gamma_2233":
        #     print('\n')
        #     pprint(gamma_dict)
        #     print('\n')
        #     exit(10)
        
    # for nn in range(3, n+1):
    #     for mm in range(2, m+1):
    # nn=n 
    # start = time.time()
    # max_eigenvals_standard = []
    # max_eigenvals_Cmxn = []
    # max_eigenvals_tensor = []
    # max_eigenvals_reg = []
    # for k in range(1):
    #     import random
    #     mean = 0.0
    #     std_dev = 1.0
    #     coefficients = [abs(np.random.normal(0,1)) for _ in range(nn)]
    #     print(coefficients)
    #     print(2*sum(coefficients[1:])**2)
    #     # random.shuffle(coefficients)
    #     for mm in range(m, m+2):
    #         group_elements, elem_to_idx, idx_to_elem = build_wreath_product(mm, nn)
    #         E_dict = coloured_octopus(coefficients, mm, nn, list(range(mm)))
    #         # max_eigenvals_standard += [max(eigenvalues_standard_sn(E_dict, nn))]
    #         print(nn,mm,eigenvalues_Cmxn(E_dict, nn, mm), max(eigenvalues_standard_sn(E_dict, nn)))
    #         print(eigenvalues_regular(E_dict, group_elements, elem_to_idx, mm)[:nn*(mm-1)+1])
            # max_eigenvals_tensor += [eigenvalues_tensor(E_dict, nn, mm)[-2:]]
            # max_eigenvals_reg += [max(eigenvalues_regular(E_dict, group_elements, elem_to_idx, mm))]
    # print(f"finished n = {nn}, m = {mm} in {time.time()-start:.2f} seconds")
    # all_eigvals = [(max_eigenvals_standard[i], max_eigenvals_Cmxn[i], max_eigenvals_tensor[i], 3000000) for i in range(len(max_eigenvals_standard))]
    # std_diff_c = 0
    # Cmn_diff_tens_c = 0
    # tens_diff_c = 0
    # Cmn_diff_reg_c = 0
    # for tup in all_eigvals:
    #     # if (tup[0] != tup[1]):
    #         # std_diff_c += 1
    #     if (tup[1][0] != tup[1][1]):
    #         print(tup[1][0], tup[1][1])
    #         Cmn_diff_tens_c += 1
        # if (tup[2] != tup[3]):
            # tens_diff_c += 1
        # if (tup[1] != tup[3]):
            # Cmn_diff_reg_c += 1
    # assert(tup[3] >= tup [2] and tup[2] >= tup[1] and tup[1] >= tup[0]), f"eigenvalues are not ordered correctly, {tup}"
    # print(f"std_diff_c = {std_diff_c}, Cmn_diff_tens_c = {Cmn_diff_tens_c}, tens_diff_c = {tens_diff_c}") #, Cmn_diff_reg_c = {Cmn_diff_reg_c}")

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
