#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SVP Challenge Solver Command Line Client
changed by CS on August 24.
实现super_workout, two-step-mode, pre-process(pnj-BKZ) and pump, check and predict the cost.
"""


import pickle as pickler


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LWE Challenge Solving Command Line Client
"""

import copy
import re
import sys
import time
import math
import os.path

# from collections import OrderedDict # noqa
from math import log

from fpylll import IntegerMatrix
# from fpylll import BKZ as fplll_bkz
# from fpylll.algorithms.bkz2 import BKZReduction
# from fpylll.tools.quality import basis_quality
from fpylll.util import gaussian_heuristic

from g6k.siever import Siever
from g6k.utils.cli import parse_args, run_all #, pop_prefixed_params
from g6k.utils.stats import dummy_tracer

from g6k.siever import SaturationError
import logging
# from fpylll.util import gaussian_heuristic

import numpy as np
np.set_printoptions(threshold=np.inf, precision=3)
from random import randint

import datetime






def load_lattice_challenge(n): #ok

    start = "lattice_challenge"

    if not os.path.isdir(start):
        print("cannot find target dir %s" % start)
        exit(1)

    end = "challenge-{n:03d}".format(n=n)
    filename = os.path.join(start, end)

    if os.path.isfile(filename) is False:
        print("file %s not exist!" % filename)
        exit(1)

    data = open(filename, "r").readlines()
    n, m, q = map(lambda x: int(x), [data[0], data[1], data[2]])

    c_index = 2 if data[3].startswith("[") else 3

    A = eval(",".join([s_.replace(" ", ", ") for s_ in data[c_index+1:]]))
    A = IntegerMatrix.from_matrix(A)
    return A, q


def sub_lattice_basis(A, n, n_shrink): #ok
    B = [[A[i][j] for j in range(n - n_shrink, n)] for i in range(n - n_shrink, n)]
    B = IntegerMatrix.from_matrix(B)
    return B


def print_pump_state(pump): #ok
    pump.minl = min(pump.g6k.l, pump.minl)
    if pump.phase != "down":
        print ("\r %3d: ↑%3d      " % (pump.r-pump.l, pump.g6k.r-pump.g6k.l))
    else:
        print ("\r %3d: ↑%3d ↓%3d " % (pump.r-pump.l, pump.r-pump.minl, pump.r-pump.g6k.l))
    sys.stdout.flush()


# calculate the gh of [l, r)
def get_gh(g6k, l, r): #ok
    return gaussian_heuristic([g6k.M.get_r(i, i) for i in range(l, min(r, g6k.full_n))])

# kappa is absolute value
def get_r0rate(g6k, kappa, blocksize): #ok
    return g6k.M.get_r(kappa, kappa) / get_gh(g6k, kappa, kappa + blocksize)



# Sieve (switching to gauss if needed) and test loop-breaking conditions
# Return true if we should continue
def wrapped_sieve(pump): #ok
    if pump.phase == "init":
        alg = "gauss"
        # alg = None
    else:
        alg = None

    dh_dim4free = min(pump.g6k.params.dh_dim4free, pump.g6k.l - pump.insert_left_bound)

    cont = True
    try:
        with pump.g6k.temp_params(saturation_ratio=pump.g6k.params.saturation_ratio * pump.sat_factor, dh_dim4free=dh_dim4free):
            # Match lifting effort to insertion strategy
            pump.g6k(alg=alg, tracer=pump.tracer)

    except SaturationError as e:
        if pump.saturation_error == "skip":
            pump.down_sieve = False
            logging.info("saturation issue: breaking pump.")
            cont = False
        elif pump.saturation_error == "weaken":
            logging.info("saturation issue: weakening pump.")
            pump.sat_factor /= 2.
            if pump.sat_factor < 0.1:
                cont = False
        elif pump.saturation_error == "ignore":
            pass
        else:
            raise e

    # if pump.phase == "up" and (pump.max_up_time is not None):
    #     if pump.max_up_time < time.time() - pump.up_time_start:
    #         cont = False

    # if pump.goal_r0 is not None:
    #     ii = pump.g6k.insert_best_lift(scoring_goal_r0, aux=pump)
    #     print("insert index: ", ii)

    # if (pump.g6k.M.get_r(pump.kappa, pump.kappa) <= pump.goal_r0):
    #     cont = False
    # print(cont)
    return cont


# def scoring_goal_r0(i, nlen, olen, aux):
#     return i == aux.kappa and nlen < aux.goal_r0

def scoring_down(i, nlen, olen, aux):
    if i < aux.insert_left_bound or nlen >= olen:
        return False
    # return log(olen / nlen) - i * log(aux.prefer_left_insert)
    return - i * log(aux.prefer_left_insert)


def pump(g6k, tracer, kappa, blocksize, dim4free, down_sieve=False, # Main parameters
         goal_r0=None, max_up_time=None, start_up_n=50, saturation_error="skip",  # Flow control of the pump
         increasing_insert_index=False, prefer_left_insert=1.2,                                     # Insertion policy
         verbose=True                                                                             # Misc
         ):
    """
    Run the pump algorithm.

    :param g6k: The g6k object to work with
    :param tracer: A tracer for g6k
    :param kappa: beginning of the block
    :param blocksize: dimension of the block (r=kappa+blocksize)
    :param dim4free: number of ``dimension for free'' [Ducas, Eurcrypt 2018]: Sieve context [l,r] where l=kappa+dim4free
    :param down_sieve: re-sieve after each insert during the pump-down phase.  (stronger reduction,
        slower running time)
    :param goal_r0: an extra hook to always insert at position kappa if this goal length can be met
        by a lift.  Quit when this is reached.
    :param max_up_time: For balancing BKZ time with SVP call time in LWE.  Stop pumping up when this
        time has elapsed.
    :param start_up_n: Initial sieve-context dimension for pumping up (starting at 1 incurs useless overheads)
    :param saturation_error: determines the behavior of pump when encountering saturation issue {"weaken",
        "skip", "ignore", "forward"}
    :param increasing_insert_index: During pump-down, always insert on the right side of the previous insertion.
    :param prefer_left_insert: Parameter theta from the paper (Sec 4.4) for scoring insertion candidates.
    :param verbose: print pump steps on the standard output.

    """

    pump.r = kappa + blocksize
    pump.l = kappa + dim4free  # noqa

    flast = blocksize

    g6k.shrink_db(0)
    g6k.lll(kappa, pump.r)
    g6k.initialize_local(kappa, max(pump.r-start_up_n, pump.l+1), pump.r)
    pump.sat_factor = 1.
    pump.up_time_start = time.time()
    pump.insert_left_bound = kappa
    pump.minl = g6k.l

    pump.saturation_ratio_down = g6k.params.saturation_ratio

    for key in ('kappa', 'down_sieve', 'goal_r0', 'g6k', 'tracer',
                'max_up_time', 'saturation_error', 'verbose', 'prefer_left_insert'):
        setattr(pump, key, locals()[key])

    goal_r0kappa = -1.0

    # pump.down_sieve = False
    with tracer.context(("pump", "kappa:%d beta:%d f:%d" % (kappa, blocksize, dim4free))):
        with g6k.temp_params(reserved_n=pump.r-pump.l, goal_r0=goal_r0kappa):
            time_start = time.time()
            if g6k.params.default_sieve == "gpu":
                pump.phase = "up"
            else:
                pump.phase = "init"
            pump.phase = "up"
            cout = wrapped_sieve(pump)  # The first initializing Sieve should always be Gauss to avoid rank-loss
            if not cout and g6k.n >= 90:
                # g6k.params.db_size_factor *= 2
                return -1

            # Pump Up
            while (g6k.l > pump.l):
                with tracer.context(("pump-step-up", "l:%d r:%d n:%d" % (g6k.l, g6k.r, g6k.n))):
                    g6k.extend_left(1)
                    print_pump_state(pump)
                    flast = g6k.l - kappa
                    cout = wrapped_sieve(pump)
                    if not cout and g6k.n >= 90:
                        g6k.db_size_factor *= 2
                        # return -1

            up_time = time.time() - time_start
            # print("\tup time: %.3f" % up_time)

            # Pump Down  
            # while (g6k.n > 3):
            #     ii = g6k.insert_best_lift(scoring_down, aux=pump)
            #     print("\tinsert index: %s" % (str(ii)))
            #     g6k.shrink_left(1)          
            g6k.resize_db(0)

            ins_T = time.time()
            ii = g6k.insert_best_lift(scoring_down, aux=pump)
            print("\tinsert index: %s" % (str(ii)))
            
            # print("total time:", time.time() - time_start)
            total_time = time.time() - time_start
            print("\tinsert time: %.3f, up time: %.3f, total time: %.3f" % (total_time - up_time, up_time, total_time))
                     
    return flast



def default_dim4free_fun(blocksize):
    """
    Return expected number of dimensions for free, from exact-SVP experiments.

    :param blocksize: the BKZ blocksize

    """
    return max(int(blocksize / log(blocksize) - 5), 0)
    # return int(11.5 + 0.075*blocksize)



def complexity_up(up):
    expo = .349
    expo = .243
    c = 0
    for i in range(50, up + 1):
        c += 2**(expo * i)
    return c

def complexity_svp(n):
    return complexity_up(n - default_dim4free_fun(n))



def complexity_bkz(d, bz):
    return (d - bz + default_dim4free_fun(bz) + 1) * complexity_svp(bz)


def complexity_overall(d, bzs, last_svp):
    cost = complexity_svp(last_svp)
    for bz in bzs:
        cost += complexity_bkz(d, bz)
    return cost

def complexity_workout_svp(n, pump_step=3, down_factor=1.5):
    final_up = n - default_dim4free_fun(n)
    # with down
    cost = complexity_up(final_up)
    for up in range(final_up - pump_step, 50, -pump_step):
        cost += complexity_up(up) * (1 + down_factor)
    return cost





def bkz_tour(g6k, tracer, blocksize):
    """
    Run a BKZ-tour: call Pump consecutively on every (jth) block.

    :param g6k: The g6k object to work with
    :param tracer: A tracer for g6k
    :param blocksize: dimension of the blocks
    :param jump: only call the pump every j blocks
    :param dim4free_fun: number of dimension for free as a function of beta (function, or string
        e.g. `lambda x: 11.5+0.075*x`)
    :param pump_params: parameters to pass to the pump
    """
    # pump_params = {"down_sieve": False}

    d = g6k.full_n
    g6k.shrink_db(0)

    Last_Mat = copy.copy(g6k.M.B)

    g6k.lll(0,d)

    if (Last_Mat == g6k.M.B):
        print("no change after lll!")

    g6k.update_gso(0,d)

    dim4free = default_dim4free_fun(blocksize)

    # indices  = [(0, blocksize - dim4free + i, i) for i in range(0, dim4free, jump)]
    indices = [(i, blocksize, dim4free) for i in range(d - blocksize)]
    indices += [(d - blocksize + i, blocksize - i, dim4free - i) for i in range(dim4free + 1)]

    start_T = time.time()

    i = 0
    while i < len(indices):
        kappa, beta, f = indices[i]

    # for i, (kappa, beta, f) in enumerate(indices):
        print("\r k:%d, b:%d, f:%d " % (kappa, beta, f))
        sys.stdout.flush()

        last_r0rate = get_r0rate(g6k, kappa, beta)

        flast = pump(g6k, tracer, kappa, beta, f, saturation_error="weaken")
        if flast == -1:
            return -1
        new_r0rate = get_r0rate(g6k, kappa, beta)

        Last_Mat = copy.copy(g6k.M.B)

        g6k.lll(0,d)

        if (Last_Mat == g6k.M.B):
            print("no change after lll!")
        # Last_Mat = copy.copy(g6k.M.B)
        i += 1
    avg_time = (time.time() - start_T) / len(indices)
    print("bkz-front-%d: average time: %.3f" % (blocksize, avg_time))

    return avg_time



def sis_kernel(arg0, params=None, seed=None):
    """
    Run the primal attack against Darmstadt SIS instance (n).

    :param n: the dimension of the SIS-challenge secret
    :param params: parameters for SIS:


        - bkz/blocksizes: given as low:high:inc perform BKZ reduction
          with blocksizes in range(low, high, inc) (after some light)
          prereduction

        - bkz/tours: the number of tours to do for each blocksize

        - bkz/jump: the number of blocks to jump in a BKZ tour after
          each pump

        - bkz/extra_dim4free: lift to indices extra_dim4free earlier in
          the lattice than the currently sieved block

        - bkz/dim4free_fun: in blocksize x, try f(x) dimensions for free,
          give as 'lambda x: f(x)', e.g. 'lambda x: 11.5 + 0.075*x'

        - pump/down_sieve: sieve after each insert in the pump-down
          phase of the pump

        - dummy_tracer: use a dummy tracer which captures less information

        - verbose: print information throughout the sis challenge attempt

    """

    # Pool.map only supports a single parameter
    if params is None and seed is None:
        n, params, seed = arg0
    else:
        n = arg0

    params = copy.copy(params)

    # params for underlying BKZ
    dim4free_fun = params.pop("bkz/dim4free_fun")

    # misc
    verbose = params.pop("verbose")


    n = 900

    n_shrink = 264
    s = 134
    for blocksize in range(60, 100, 2):
    # for blocksize in range(110, 120, 2):

        A, q = load_lattice_challenge(n)


        # 需要严格比目标小
        print("test solving sis challenge-%d" % n)
        print(datetime.datetime.now())
        target_norm = q * q
        print("target norm: %d" % target_norm)
        
        print("Chose %d sub latiice. Process pump-%s for 10 times" % (n_shrink, str(blocksize)))

        B = sub_lattice_basis(A, n, n_shrink)

        g6k = Siever(B, params)
        print("after initialization: ")
        print(datetime.datetime.now())
        
        tracer = dummy_tracer

        d = g6k.full_n
        g6k.lll(0, g6k.full_n)

        T0 = time.time()
        T0_BKZ = time.time()

        avg = []

        print([g6k.M.get_r(i,i) for i in range(d)])

        repeat_time = 1

        # BKZ
        block_index = 0
        while(block_index < repeat_time):
        # for blocksize in blocksizes:
  
            bkz_start_T = time.time()

            print ("Starting a BKZ-%d tour. " % (blocksize))        
            avg_time = bkz_tour(g6k, tracer, blocksize)
            if avg_time == -1:
                print("ERROR! BKZ tour failed due to saturation issue. Restart with larger db_size_factor.")
            st = "bkz-%d, tour walltime: %.3f sec, avg time: %.3f" % (blocksize, time.time() - bkz_start_T, avg_time)
            print(st)

            avg += [(blocksize, time.time() - bkz_start_T, avg_time, complexity_bkz(d, blocksize), complexity_svp(blocksize))]
            
            print("bkz sofar time: %.3f sec " % (time.time() - T0_BKZ))
            block_index += 1
            print([g6k.M.get_r(i,i) for i in range(d)])

        T_BKZ = time.time() - T0_BKZ
        print([g6k.M.get_r(i, i) for i in range(10)])


        # for it in avg:
        #     print("\tbkz-%d, up %d, total time: %.3f, avg time: %.3f, est_c: %.3f, c./t.: %.3f" % (it[0], it[0] - default_dim4free_fun(it[0]), it[1], it[2], it[4], it[4]/it[2]))

        # T_SVP = time.time() - T0 - T_BKZ
        # print("\tsvp-%d, up %d, time: %.3f sec, est_c: %.3f, c./t.: %.3f" % (s, s - default_dim4free_fun(s), T_SVP, complexity_svp(s), complexity_svp(s) / T_SVP))
        # print("\t all walltime: %.3f sec" % (time.time() - T0))




def sis():
    """
    Attempt to solve an lwe challenge.

    """
    description = sis.__doc__

    args, all_params = parse_args(description)

    stats = run_all(sis_kernel, all_params.values(), # noqa
                    lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    step_size=args.step_size,
                    trials=args.trials,
                    workers=args.workers,
                    seed=args.seed)


if __name__ == '__main__':
    sis()

# python3 test_pump_time.py 900 --threads 32 --gpus 1 2>&1 | tee logs/Test_Pump_Average_Time_900.log