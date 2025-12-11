from sage.all import log, exp
from sage.all import line, save, load, identity_matrix, matrix
from sage.all import RealDistribution, prod, erf
from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ
from BKZ_Simulator import simulate as CN11_simulate
from BKZ_Simulator import simulate_prob as BSW18_simulate
from load_lwechal import load_lwe_challenge
import BSW18

import math, copy, os

import time


###########################################################

version = "ver. 2.1, completely automatic"




gap = 25
bkz_factor_exp = 1



import logging


def setup_logging(n, alpha):
    # 日志格式（时间-日志级别-消息）
    # log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    log_format = "%(message)s"


    # 配置文件处理器（追加模式，utf-8 编码）
    log_name = f"log_Strategy/log_LWE_n{n}_alpha{alpha}"
    file_handler = logging.FileHandler(log_name, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format))
    

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(logging.Formatter(log_format))
    logging.root.handlers = []

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

###########################################################


chisquared_table = {i: None for i in range(1025)}

for i in range(1025):
    chisquared_table[i] = RealDistribution('chisquared', i)


############ matrix basics ############
def primal_lattice_basis(A, c, q, m=None):
    """
    Construct primal lattice basis for LWE challenge
    ``(A,c)`` defined modulo ``q``.

    :param A: LWE matrix
    :param c: LWE vector
    :param q: integer modulus
    :param m: number of samples to use (``None`` means all)

    """
    if m is None:
        m = A.nrows
    elif m > A.nrows:
        raise ValueError("Only m=%d samples available." % A.nrows)
    n = A.ncols

    B = IntegerMatrix(m+n+1, m+1)
    for i in range(m):
        for j in range(n):
            B[j, i] = A[i, j]
        B[i+n, i] = q
        B[-1, i] = c[i]
    B[-1, -1] = 1

    B = LLL.reduction(B)
    assert(B[:n] == IntegerMatrix(n, m+1))
    B = B[n:]

    return B

# def gen_lwechal_instance(n=40, alpha=0.005, default_g6k = False):
#     A, c, q = load_lwe_challenge(n=n, alpha=alpha)
    
#     logging.info("-------------------------")
#     logging.info("Primal attack, TU LWE challenge n=%d, alpha=%.4f, q = %d. " % (n, alpha, q))

#     try:
#         min_cost_param = gsa_params(n=A.ncols, alpha=alpha, q=q, decouple=True)
#         (b, s, m) = min_cost_param
#     except TypeError:
#         raise TypeError("No winning parameters.")
   
#     logging.info("Chose %d samples. Predict solution at bkz-%d + svp-%d." % (m, b, s))
    
#     d = m + 1

#     B = primal_lattice_basis(A, c, q, m=m)

#     sigma = alpha * q
    
#     M = GSO.Mat(B)
#     M.update_gso()
#     rr = [M.get_r(i,i) for i in range(d)]
#     if(default_g6k):
#         log_rr = [log(rr[i],2)/2. for i in range(d)]
#     else:
#         log_rr = [log(rr[i],2)/2. - log(sigma,2) for i in range(d)]
#         sigma = 0.
#     logging.info(f"Initial slope: {get_current_slope(log_rr,0,d)}")
    
#     dvol = sum(log_rr) * log(2)  #ln(vol)

#     dim = m + 1
#     logging.info("dim = %3d, dvol = %3.7f" %(dim, dvol))
#     print()

#     # return (dim, dvol)
#     return (log_rr,dim,dvol,b,sigma)

# return full of tar_dim, q
def load_whole_matrix(tar_dim):
    src_file = "official/challenge-%d" % (tar_dim)

    f = open(src_file, "r")
    all_lines = f.readlines()
    f.close()

    assert(all_lines[0] == "%d\n" % tar_dim)
    q = int(all_lines[1].replace("\n", ""))
    assert(all_lines[2] == "%d\n" % q)

    full = []
    for row in range(tar_dim):

        line = all_lines[row + 3].replace("[", "").replace("]", "").replace("\n", "")
        lsp = line.split(" ")

        assert(len(lsp) == tar_dim)

        full += [[int(c) for c in lsp]]
    logging.info("full matrix with dim %d is loaded, q = %d" % (tar_dim, q))

    return full, q

# return res
def matrix_shrink(full, shrink_to_dim):
    d = full.nrows
    res = []
    for row in range(d - shrink_to_dim, d):
        # assert(sum([c*c for c in full[row][:d - shrink_to_dim]]) == 0)
        res = [full[row][d - shrink_to_dim:]] + res
    logging.info("matrix has been shrinked from %d to %d" % (len(full), shrink_to_dim))
    return res

# return the squared values of the gso basis.
def gso_init(shrink_mat, sigma):

    FPLLL.set_random_seed(1337)
    # mat = IntegerMatrix.from_matrix(shrink_mat)
    # logging.info("converted matrix to IntegerMatrix")

    # A = LLL.reduction(shrink_mat)
    # M = GSO.Mat(A)
    # M.update_gso()
    M = GSO.Mat(shrink_mat, float_type="mpfr", flags=GSO.ROW_EXPO)
    M.update_gso()
    logging.info("GSO basis is computed")
    profile = [M.get_r(i, i)/sigma/sigma for i in range(shrink_mat.nrows)]
    for (i, r) in enumerate(profile):
        if r <= 0:
            profile[i] = 1e-20
    return profile


############ complexity basics ############

def ball_log_vol(n):
    """
    Return volume of `n`-dimensional unit ball
    :param n: dimension
    """
    return (n/2.) * math.log(math.pi) - math.lgamma(n/2. + 1)

def gaussian_heuristic(r2list):
    """
    Return squared norm of shortest vector as predicted by the Gaussian heuristic.
    :param r: vector of squared Gram-Schmidt norms
    """
    n = len(list(r2list))
    log_vol = sum([math.log(x) for x in r2list])
    log_gh =  1./n * (log_vol - 2 * ball_log_vol(n))
    return math.exp(log_gh)

def get_targh_svp(r2list, last_svp, target):    
    return target / gaussian_heuristic(r2list[:last_svp])

def dim4free_n_div_logn(n):
    if n < 50:
        return 0
    dim4free = int(n / math.log(n))
    return int(min((n - 40) / 2, dim4free))



expo1 = .202
expo2 = .249
expo3 = .296
expo = .349



def complexity_sieve(n):
    max_up = n - (n / math.log(n)) + 5
    # logging.info("up: ", max_up)
    global expo1
    global expo2
    global expo3
    eexpo = 0
    if max_up <= 80:
        eexpo = expo1
        return 2**(eexpo * max_up)

    if max_up >80 and max_up <= 100:
        eexpo = expo2
        return 2**(eexpo * max_up)

    if max_up > 100 and max_up <= 140:
        eexpo = expo3
        return 2**(eexpo * max_up)

    if max_up > 140:
        eexpo = 0.349
        return 2**(eexpo * max_up)

def complexity_svp(n): # ver2.1
    max_up = n - (n / math.log(n))
    global expo
    c = 0
    for ii in range(int(max_up)):        
        c += 2**(expo * (max_up - ii))
    return c    

def complexity_bkz(d, bz):
    return max(1, d - bz + 1) * complexity_sieve(bz)

def complexity_overall(d, bzs, last_svp, bkz_factor=1.0):
    cost = complexity_sieve(last_svp)
    for bz in bzs:
        cost += complexity_bkz(d, bz) * bkz_factor
    return cost



############ simulator ############


# r is squared values of gso
def sim_bkz(r, block_size): # ok2.1b
    # logging.info("simulate bkz %d" % block_size)
    cn11 = CN11_simulate(r, BKZ.Param(block_size=block_size, max_loops=1))
    return cn11[0]


# return overall cost
def apply_bkz(ori_r, bzs, last_svp, remaining_proba, verbose=True):
    if verbose:
        logging.info(f"applying: bzs = {bzs} , last_svp = {last_svp}")
    cur_r = copy.copy(ori_r)
    rs = []
    i = 0
    cumulated_proba = 1. -  remaining_proba

    for bz in bzs:
        last_r0 = cur_r[0]
        cur_r = sim_bkz(cur_r, bz)
        # assert(cur_r[0] > lastr0 * 0.999)
        rs += [copy.copy(cur_r)]
        i += 1
        proba = 1. * success_probability(cur_r, last_svp)
        cumulated_proba += remaining_proba * proba
        remaining_proba = 1. - cumulated_proba
        if verbose:
            logging.info("%d: bkz %d remaining pro: %f" % (i, bz, remaining_proba))
            # gh = gaussian_heuristic(cur_r[-last_svp:])
            # logging.info(f"gh(L({len(cur_r)-last_svp}:{len(cur_r)})): {gh}")
    cost = complexity_overall(len(cur_r), bzs, last_svp)
    if verbose:
        logging.info(f"done, overall cost {cost}")
    if remaining_proba > .001:
        if verbose:
            logging.info("fail!!!!!\n\n")
        return -1, remaining_proba
    else:
        if verbose:
            logging.info("success~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
        return cost, remaining_proba


#################################### my work ####################################


def reduced_bz_num(ori_r, bz):
    cost = 0
    cur_r = copy.copy(ori_r)
    # logging.info(cur_r[0])
    i = 0
    
    while True:
        last_r = copy.copy(cur_r)
        cur_r = sim_bkz(cur_r, bz)
        i += 1
        cost += complexity_bkz(len(ori_r), bz)
        # logging.info(cur_r[0])
        if cur_r == last_r:
            break
    return i, cost

def success_probability(ori_r, beta):

    i = len(ori_r) - beta
    proba = 1.
    gh = gaussian_heuristic(ori_r[-beta:])
    # logging.info(f"gh(L({len(ori_r)-beta}:{len(ori_r)})): {gh}")
    proba *= chisquared_table[beta].cum_distribution_function(gh)
    # logging.info(f"bz: {beta}, i: {i}, ori_r_i: {ori_r[i]}, success pro: {proba}")
    return proba

# return the number of bz to succeed
def succ_bz_num(ori_r, bz, last_svp, remaining_proba):
    # logging.info(r0rate)
    last_r0rate = 0
    last_proba = 2
    cur_r = copy.copy(ori_r)
    i = 0

    cumulated_proba = 0.

    while remaining_proba > .001:
        last_r0 = cur_r[0]
        cur_r = sim_bkz(cur_r, bz)
        # logging.info(cur_r)
        # logging.info(cur_r[0], lastr0 * 0.999, cur_r[0] > lastr0 * 0.999, bz, last_svp)
        # assert(cur_r[0] > lastr0 * 0.999)
        i += 1
        proba = 1. * success_probability(cur_r, last_svp)
        cumulated_proba += remaining_proba * proba
        remaining_proba = 1. - cumulated_proba
        # logging.info()
        # logging.info(f"cumulated_proba: {cumulated_proba}, remaining proba: {remaining_proba}")
        # gh = gaussian_heuristic(cur_r[-last_svp:])
        # logging.info(f"gh(L({len(cur_r)-last_svp}:{len(cur_r)})): {gh}")
        # exit(1)
        
        # if cur_r[0] > 1.001 * last_r0:
        #     # logging.info(r0rate, 1.001 * last_r0rate, r0rate < 1.001 * last_r0rate)
        #     return -1
        # if last_proba == proba:
        if abs(last_proba - proba) < 1e-10:
            logging.info(f"{i},  bz: {bz}, remaining_proba: {remaining_proba}")
            return -1
        if i >= 100:
            logging.info("error too many iter!")
            logging.info(f"{i},  bz: {bz}, remaining_proba: {remaining_proba}")
        last_proba = proba
    return i

def best_bkz_only(ori_r, max_svp):
    step_sizes = [64, 16, 4, 1]
    rng = [(max_svp) % 64 + 64, (max_svp)]

    # find the minimal success one
    for step in step_sizes:
        bz = rng[0]
        n_bz = succ_bz_num(ori_r, bz, bz, 1.)
        while bz <= rng[-1] and n_bz < 0:
            bz += step
            if bz <= rng[-1]:
                n_bz = succ_bz_num(ori_r, bz, bz, 1.)
            else:
                n_bz = -1

        if bz > rng[-1]: # cannot success
            return bz, -1, 0
        # bz succeed

        rng = [max(45, bz - step), bz] # print("find new range", rng)


    cost = complexity_bkz(len(ori_r), bz) * (n_bz+1)
    bzs = n_bz + 1
    # print(bzs)
    logging.info("bz = %d, num = %d, cost = %.0f" % (bz, bzs, cost))
    # logging.info("last svp = %d, num = %d, cost = %.0f, svp cost = %.0f" % (last_svp, len(bzs), cost, complexity_svp(last_svp)))
                                                                                                                                          
    return bz, bzs, cost  

def best_succ_bz(ori_r, last_svp):
    step_sizes = [64, 16, 4, 1]
    rng = [(last_svp - 0) % 64 + 64, (last_svp - 0)]

    # find the minimal success one
    for step in step_sizes:
        bz = rng[0]
        n_bz = succ_bz_num(ori_r, bz, last_svp, 1.)
        while bz <= rng[-1] and n_bz < 0:
            bz += step
            if bz <= rng[-1]:
                n_bz = succ_bz_num(ori_r, bz, last_svp, 1.)
            else:
                n_bz = -1

        if bz > rng[-1]: # cannot success
            return [], -1
        # bz succeed

        rng = [max(45, bz - step), bz] # print("find new range", rng)

    pair = []
    global bkz_factor_exp
    # print(f"ttbz: {bz}")
    for ttbz in range(bz, min(bz + 10, last_svp + 1)):
        
        n_ttbz = succ_bz_num(ori_r, ttbz, last_svp, 1.)
        # print(f"ttbz: {ttbz}, n: {n_ttbz}")
        if n_ttbz > 0:
            bzs = [ttbz] * n_ttbz
            pair += [(complexity_overall(len(ori_r), bzs, last_svp, bkz_factor=bkz_factor_exp), bzs)]
        # print(pair[-1])
    pair.sort()
    cost = pair[0][0]
    bzs = pair[0][-1]
    # print(bzs)
    logging.info("last svp = %d, bz = %d, num = %d, cost = %.0f, svp cost = %.0f" % (last_svp, bzs[0], len(bzs), cost, complexity_sieve(last_svp)))
    # print("last svp = %d, num = %d, cost = %.0f, svp cost = %.0f" % (last_svp, len(bzs), cost, complexity_svp(last_svp)))

    return bzs, cost


# return result like (7083793150306219.0, [148, 148, 148], 176)
def get_best_lastsvp(ori_r, max_svp, min_svp):
    res = []
    
    logging.info("\ntest last_svp in range: %d ~ %d" % (min_svp, max_svp))
    for last_svp in range(min_svp, max_svp+1):
        bzs, cost = best_succ_bz(ori_r, last_svp)
        if len(bzs) > 0:
            res += [(cost, bzs, last_svp)]

            if complexity_sieve(last_svp) > 1.5 * cost:
                break

    if len(res) == 0:
        logging.info("the target cannot be solved by shrinking to %d\n\n" % (len(ori_r)))
        return (-1, [], -1)

    res.sort()
    logging.info(f"best , {res[0]}, \n")

    return res[0]

def get_best_d(n, alpha, min_d, max_d, min_svp, max_svp):
    # full, q =  load_whole_matrix(tar_dim)
    A, c, q = load_lwe_challenge(n=n, alpha=alpha)
    best_res = None
    best_d = -1
    sigma = alpha * q

    for d in range(min_d, max_d, 5):
        logging.info("test shrink dim %d" % d)
        B = primal_lattice_basis(A, c, q, m=d-1)
        # mat_shrink = matrix_shrink(A, d)
        lll_r = gso_init(B, sigma)

        res = get_best_lastsvp(lll_r, max_svp, min_svp)

        if res[0] > 0:
            if best_d == -1 or best_res[0] > res[0]:
                best_d = d
                best_res = res

        if best_d > 0 and d >= best_d + 10:
            break


    if best_d == -1:
        logging.info("cannot find best d in range [250, 400)")
        exit(1)
    else:
        logging.info("find rough best d at %d, with result %s" % (best_d, str(best_res)))

    for d in range(best_d - 5 + 1, best_d + 5):

        logging.info("test shrink dim %d" % d)
        B = primal_lattice_basis(A, c, q, m=d-1)
        lll_r = gso_init(B, sigma)
        # mat_shrink = matrix_shrink(A, d)
        # lll_r = gso_init(mat_shrink)
        res = get_best_lastsvp(lll_r, max_svp, min_svp)

        if res[0] > 0:
            if best_res[0] > res[0]:
                best_d = d
                best_res = res

    logging.info("find final best d at %d, with result (factored cost) %s" % (best_d, str(best_res)))

    return best_d, best_res, q




# return the number of bz to succeed
# def succ_bz_num_suffix(ori_r, bz, suffix, last_svp):
#     max_times = 15
#     # max_cost, _ = apply_bkz(ori_r, [bz] * max_times + suffix, last_svp, 1., verbose=False)
#     # if max_cost < 0:
#     #     return (-1, -1)
#     remaining_proba = 1.
#     for i in range(1, max_times):
#         cost, remaining_proba = apply_bkz(ori_r, [bz] * i + suffix, last_svp, remaining_proba, verbose=False)
#         if cost > 0:
#             return (cost, i)
#     return (-1, -1)
#     return (max_cost, max_times)


def success_probability_suffix(ori_r, suffix, beta, remaining_proba):
    cur_r = copy.copy(ori_r)

    cumulated_proba = 1. -  remaining_proba

    for bz in suffix:
        cur_r = sim_bkz(cur_r, bz)
        # assert(cur_r[0] > lastr0 * 0.999)

        proba = 1. * success_probability(cur_r, beta)
        cumulated_proba += remaining_proba * proba
        remaining_proba = 1. - cumulated_proba
    # logging.info(f"success_probability_suffix: ", remaining_proba)
    return remaining_proba

def succ_bz_num_suffix(ori_r, bz, suffix, last_svp):
    remaining_proba = 1.
    last_r0rate = 0
    last_proba = 1.1
    cur_r = copy.copy(ori_r)
    i = 0

    cumulated_proba = 0.

    while remaining_proba > .001:

        last_r0 = cur_r[0]
        cur_r = sim_bkz(cur_r, bz)

        i += 1
        proba = 1. * success_probability(cur_r, last_svp)
        cumulated_proba += remaining_proba * proba
        remaining_proba = 1. - cumulated_proba
        if success_probability_suffix(cur_r, suffix, last_svp, remaining_proba) <= .001:
            return i
        # if last_proba == proba:
        if abs(last_proba - proba) < 1e-10:
            logging.info(f"bz; {bz}, i: {i}, remaining_proba: {remaining_proba}")
            return -1
        last_proba = proba
    return i

# best (cost, bz, n_bz) with + suffix + last_svp
def best_succ_bz_suffix(ori_r, suffix, last_svp):
    assert(len(suffix) > 0)

    ggg = 3
    local_strategy = []
    for bz in range(max(45, suffix[0] - ggg), suffix[0]):
        start = time.time()
        n_bz = succ_bz_num_suffix(ori_r, bz, suffix, last_svp)
        end = time.time()
        logging.info(f"one bz time: {end-start}")
        if n_bz > 0:
            return (complexity_overall(len(ori_r), n_bz*[bz]+suffix, last_svp), bz, n_bz) #1111

    if len(local_strategy) == 0:
        return (-1, -1, -1)


def get_best_bzs(n, alpha, d, cost_bound, init_bzs, last_svp, q, expect_d, expect_svp, strategy):    

    # r_file_name = "lwechallenge/%d-%f-challenge-profile.txt" % (n, alpha)
    # if os.path.exists(r_file_name):
    #     f = open(r_file_name, "r")
    #     all_lines = f.readlines()
    #     f.close()
    #     lll_r = [float(line.replace("\n", "")) for line in all_lines]
    # else:
    #     # full, q =  load_whole_matrix(tar_dim)
    #     A, c, q = load_lwe_challenge(n=n, alpha=alpha)
    #     # mat_shrink = matrix_shrink(full, d)
    #     # lll_r = gso_init(mat_shrink)
    #     B = primal_lattice_basis(A, c, q, m=d-1)
    #     lll_r = gso_init(B, sigma)
    #     f = open(r_file_name, "w")
    #     for i in range(len(lll_r)):
    #         f.write(str(lll_r[i]) + "\n")
    #     f.close()

    start = time.time()
    A, c, q = load_lwe_challenge(n=n, alpha=alpha)
    sigma = alpha * q
    # mat_shrink = matrix_shrink(full, d)
    # lll_r = gso_init(mat_shrink)
    B = primal_lattice_basis(A, c, q, m=d-1)
    lll_r = gso_init(B, sigma)

    ####
    global gap
    logging.info(f"let's start with (factored cost) , {cost_bound}, {init_bzs}, {last_svp}")
    all_strategy = [] # (cost, bzs, last_svp, d) # cost_bound is factored, theoretical
    # suffixes = [[bz] for bz in range(init_bzs[0] + gap + 1, init_bzs[0] - 2, -1)]
    suffixes = [[init_bzs[0]]] + [[bz] for bz in range(init_bzs[0] + gap, init_bzs[0], -1)] + [[init_bzs[0] - 1]]

    r = 0
    best_real_cost = -1
    while len(suffixes) > 0:
        logging.info(f"======== round {r} ========\ntodo suffixes: {suffixes}")
        r += 1
        cur_suffix = suffixes[0]
        suffixes = suffixes[1:]
        logging.info("cur_suffix: {cur_suffix}")

        if best_real_cost > 0 and complexity_overall(d, cur_suffix, last_svp) >= best_real_cost:
            logging.info("the suffix is too costly, continue\n\n")
            continue

        (cost, bz, n_bz) = best_succ_bz_suffix(lll_r, cur_suffix, last_svp)

        logging.info(f"fisrt choice: ({cost}, {bz}, {n_bz})")
        logging.info(f"baseline:  ({cost_bound}, {init_bzs}, {last_svp}, {d})")

        if cost < 0:
            logging.info("the strategy is infeasible, continue\n\n")
            continue

        if bz < 60 or bz < cur_suffix[-1] - 20:
            all_strategy += [(cost, [bz]*n_bz + cur_suffix, last_svp, d)]
        else:
            for newbz in range(bz, min(bz + gap, cur_suffix[0] - 1) + 1):
                suffixes = [[newbz] + cur_suffix] + suffixes

        all_strategy.sort()
        all_strategy = all_strategy[:5]
        logging.info("best 5 choices:")
        for strat in all_strategy:
            logging.info(strat)
        if len(all_strategy) > 0:
            best_real_cost = all_strategy[0][0]

        logging.info("end round with %d suffixes\n\n" % (len(suffixes)))
        

    all_strategy.sort()
    end = time.time()
    logging.info("strategy search time: %.2f seconds" % (end - start))
    if len(all_strategy) > 0:

        logging.info("-------our strategy.-------")
        logging.info(all_strategy[0])
        logging.info(float(log(all_strategy[0][0], 2)))
        best_S = all_strategy[0]


        logging.info("-------BKZ+SVP.----------")
        bzs = succ_bz_num(lll_r, best_S[1][-1], best_S[2], 1.)
        cost = complexity_overall(best_S[3], [best_S[1][-1]] * bzs, best_S[2]) 
        logging.info(f"{bzs} * {best_S[1][-1]} + {best_S[2]}")
        logging.info(float(log(cost, 2)))


        logging.info("------ BKZ-only.---------")
        min_svp = expect_svp - 30
        max_svp = expect_svp + 30
        bz, bzs, cost = best_bkz_only(lll_r, max_svp)
        logging.info(f"{bzs} *  bz")
        logging.info(float(log(cost, 2)))

        logging.info("------ XWW-bkz.----------")
        logging.info(apply_bkz(lll_r, strategy, expect_svp, 1))
        logging.info(strategy)
        cost = complexity_overall(expect_d, strategy, expect_svp)
        logging.info(float(log(cost, 2)))


def SIS_startegy_search(n, alpha, expect_d, expect_svp, strategy):

    min_d = expect_d - 20
    max_d = expect_d + 20

    min_svp = expect_svp - 20
    max_svp = expect_svp + 20

    start = time.time()

    best_d, best_res, q = get_best_d(n, alpha, min_d, max_d, min_svp, max_svp)

    get_best_bzs(n, alpha, best_d, best_res[0], best_res[1], best_res[2], q, expect_d, expect_svp, strategy)
    end = time.time()
    logging.info("total time: %.2f seconds" % (end - start))


n = 80
alpha = 0.005
 
expect_d = 271
expect_svp = 162
strategy = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]
setup_logging(n, alpha)
SIS_startegy_search(n, alpha, expect_d, expect_svp, strategy)



n = 90
alpha = 0.005

expect_d = 307
expect_svp = 177
strategy = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]


setup_logging(n, alpha)
SIS_startegy_search(n, alpha, expect_d, expect_svp, strategy)

exit(1)

n = 100
alpha = 0.005

expect_d = 351
expect_svp = 203
strategy = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158]
setup_logging(n, alpha)
SIS_startegy_search(n, alpha, expect_d, expect_svp, strategy)

n = 70
alpha = 0.01

expect_d = 263
expect_svp = 174
strategy = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123]

setup_logging(n, alpha)
SIS_startegy_search(n, alpha, expect_d, expect_svp, strategy)

n = 80
alpha = 0.01

expect_d = 306
expect_svp = 206
strategy = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161]

setup_logging(n, alpha)
SIS_startegy_search(n, alpha, expect_d, expect_svp, strategy)


n = 60
alpha = 0.015
A, c, q = load_lwe_challenge(n=n, alpha=alpha)

expect_d = 242
expect_svp = 174
strategy = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123]

setup_logging(n, alpha)
SIS_startegy_search(n, alpha, expect_d, expect_svp, strategy)


n = 70
alpha = 0.015

expect_d = 285
expect_svp = 212
strategy = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165]

setup_logging(n, alpha)
SIS_startegy_search(n, alpha, expect_d, expect_svp, strategy)
