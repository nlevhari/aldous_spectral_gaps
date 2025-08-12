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
import signal

def run_single_element_check(n, m, E_dict, c_dict_E, verbose=False):
    def myprint(*args):
        if verbose:
            print(args)
    
    # 2a) Regular representation
    group_elements, elem_to_idx, _ = build_wreath_product(n=n, m=m)
    vals_reg = eigenvalues_regular(E_dict, group_elements, elem_to_idx, m)
    # second smallest
    
    myprint("Regular representation: smallest two eigenvalues =",
          vals_reg[-1], vals_reg[-2])
    
    # 2b) Representation on (C_m x [n])
    vals_cmxn = eigenvalues_Cmxn(E_dict, n, m)
    
    myprint("C_m x [n] representation: smallest two eigenvalues =",
          vals_cmxn[-1], vals_cmxn[-2])
    
    #2c) Representation on [n]
    vals_n = eigenvalues_standard_sn(E_dict, n)
    myprint("[n] representation: smallest two eigenvalues =", 
          vals_n[-1], vals_n[-2])
    # print("[n]:", vals_n)
    # print("Cmx[n]:", vals_cmxn)
    # print("regular:", vals_reg)
    should_be_non_positive = [vals_reg[-1], vals_cmxn[-1], vals_n[-1], vals_reg[-2] - vals_cmxn[-2], vals_cmxn[-2] - vals_n[-2]]
    if (max(should_be_non_positive) > 0.0001):
        print("bugggggg")
        print("[vals_reg[-1], vals_cmxn[-1], vals_n[-1], vals_reg[-2] - vals_cmxn[-2], vals_cmxn[-2] - vals_n[-2]]=\n",should_be_non_positive)
        print("c values:", c_dict_E)
        exit(1)
    elif vals_cmxn[-2] - vals_reg[-2] > 0.001:
        print(f"diff between Cmx[n] rep and regular: {vals_cmxn[-2] - vals_reg[-2]}")
        print("[n]:", vals_n[-2])
        print("Cmx[n]:", vals_cmxn[-2])
        print("regular:", vals_reg[-2])
        print("c values:", c_dict_E)
    elif vals_n[-2] - vals_reg[-2] > 0.001:
        print(f"diff between [n] rep and regular: {vals_n[-2] - vals_reg[-2]}")
        print("[n]:", vals_n[-2])
        print("Cmx[n]:", vals_cmxn[-2])
        print("regular:", vals_reg[-2])
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

def check_Gammas(n,m):
    group_elements, elem_to_idx, _ = build_wreath_product(n=n, m=m)
    for gamma in Gammas:
        gamma_dict = gamma[0](m,n)
        e_vals_regular = eigenvalues_regular(gamma_dict, group_elements, elem_to_idx, m)
        e_vals_Cmxn = eigenvalues_Cmxn(gamma_dict, n, m)
        e_vals_std = eigenvalues_standard_sn(gamma_dict, n)
        print(f"first mn eigenvals for {gamma[1]}:", e_vals_regular[: m * n])
        print(f"Cmxn eigenvals for {gamma[1]}:", e_vals_Cmxn)
        print(f"std eigenvals for {gamma[1]}:", e_vals_std)
    
def wrap_reg_eigenvals_to_not_take_too_long(E_element, group_elements, elem_to_idx, m):
    """
    A wrapper to compute eigenvalues of the regular representation of E_element
    but with a timeout to prevent long computations.
    """
    try:
        class TimeoutException(Exception):
            pass

        def timeout_handler(signum, frame):
            raise TimeoutException

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)  # Set timeout to 2 minutes (120 seconds)

        eigenvals = eigenvalues_regular(E_element, group_elements, elem_to_idx, m)

        signal.alarm(0)  # Disable the alarm after successful execution
        return eigenvals
    except TimeoutException:
        print("eigenvalues_regular computation timed out. Returning [0, 0, 9999, 9999].")
        return [0, 0, 9999, 9999]
    
def check_extreme_eigvals(eigenvals_standard, eigenvals_Cmxn, eigenvals_tensor, eigenvals_reg, max=False):
    if max:
        r = range(-1, 0)
        comp_func = lambda x, y: x > y
    else: 
        r = range(0, 2)
        comp_func = lambda x, y: x < y
    all_eigvals = [(eigenvals_standard[i], eigenvals_Cmxn[i], eigenvals_tensor[i], eigenvals_reg[i]) for i in r]
    std_diff_c = 0
    Cmn_diff_tens_c = 0
    tens_diff_c = 0
    Cmn_diff_reg_c = 0
    for i in range(len(all_eigvals)):
        tup = all_eigvals[i]
        for j in range(len(tup) - 1):
            if comp_func(tup[j], tup[j+1]):
                print(f"{j}: eigenvalues not ordered correctly: {tup}")
                exit(1)
        std_diff_c = abs(tup[0] - tup[1])
        Cmn_diff_tens_c = abs(tup[1] - tup[2])
        if Cmn_diff_tens_c > 0.001:
            print(f"diff between Cmx[n] and tensor: {Cmn_diff_tens_c}")
            print("Cmx[n]:", tup[1])
            print("tensor:", tup[2])
            exit(1)
        tens_diff_c = abs(tup[2] - tup[3])
        Cmn_diff_reg_c = abs(tup[1] - tup[3])
        # print(f"{i}: {tup}; std_diff_c = {std_diff_c}, Cmn_diff_tens_c = {Cmn_diff_tens_c}, tens_diff_c = {tens_diff_c}, Cmn_diff_reg_c = {Cmn_diff_reg_c}")


def check_octopus(n, m, check_min=True, check_max=False):
    for nn in range(3, n+1):
        for mm in range(2, m+1):
            for k in range(10):
                group_elements, elem_to_idx, _ = build_wreath_product(n=nn, m=mm)
                start = time.time()
                coefficients = [abs(np.random.normal(0,1)) for _ in range(nn)]
                E = coloured_octopus(coefficients, m=mm, n=nn, G_elems=list(range(mm)))
                eigenvals_standard = eigenvalues_standard_sn(E, nn)
                eigenvals_Cmxn = eigenvalues_Cmxn(E, nn, mm)
                eigenvals_tensor = eigenvalues_tensor(E, nn, mm)
                eigenvals_reg = wrap_reg_eigenvals_to_not_take_too_long(E, group_elements, elem_to_idx, mm)
                print(f"finished n = {nn}, m = {mm} in {time.time()-start:.2f} seconds")
                if check_min:
                    check_extreme_eigvals(eigenvals_standard, eigenvals_Cmxn, eigenvals_tensor, eigenvals_reg, max=False)
                if check_max:
                    check_extreme_eigvals(eigenvals_standard, eigenvals_Cmxn, eigenvals_tensor, eigenvals_reg, max=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", default=3, help="size of the action set for the symmetric group", type=int)
    p.add_argument("-m", default=2, help="order of the cyclic group", type=int)
    p.add_argument("-r", "--randomize-c", help="flag if one wants to randomize the coefficients with a normal distribution", action="store_true")
    p.add_argument("-s", "--stats", help="flag if one wants to run randomizations of c vectors and check if the spectral gaps are the same", action="store_true")
    p.add_argument("--single", help="flag if one wants to run a single check", action="store_true")
    p.add_argument("-nr", "--num-randomizations", default=5, type=int,)
    p.add_argument("-c",
                    "--c-dict", default=None)
    p.add_argument("-co", "--check-octopus", action="store_true")
    p.add_argument("-com", "--check-octopus-max", action="store_true",)
    p.add_argument("-cg", "--check-gammas", action="store_true")
    args = p.parse_args()
    c_dict_ = None
    if args.c_dict is not None:
        with open(args.c_dict, 'r') as f:
            raw = json.load(f)
            c_dict_ = {tuple(map(int, k.strip("()").split(", "))): v for k, v in raw.items()}
    
    if args.check_octopus or args.check_octopus_max:
        check_octopus(args.n, args.m, check_min=args.check_octopus, check_max=args.check_octopus_max)
    if args.check_gammas:
        check_Gammas(args.n, args.m)
    if args.stats:
        for nn in range(3, args.n+1):
            for mm in range(2, args.m+1):
                print(f"running on n = {nn} and m = {mm}")
                start = time.time()
                for i in range(args.num_randomizations):
                    eigenvalue_computation(nn, mm, randomize_c=True)
                print("took: ", time.time()-start,"s")
    if args.single:
        eigenvalue_computation(args.n, args.m, randomize_c=args.randomize_c, verbose=True, c_dict_=c_dict_)
