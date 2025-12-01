from sage.all import log, exp, RR, sqrt, floor
from sage.all import line, save, load, identity_matrix, matrix
from sage.all import RealDistribution, prod, erf
from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ
from BKZ_Simulator import simulate as CN11_simulate
from BKZ_Simulator import simulate_prob as BSW18_simulate

import math, copy, os


###########################################################

version = "ver. 2.1, completely automatic"

expo1 = .202
expo2 = .249
expo3 = .296
expo = .349



# Dilithium 2
expect_d = 2304
expect_svp = 453



# Dilithium 3
expect_d = 3052
expect_svp = 721


# Dilithium 5
expect_d = 4081
expect_svp = 1040


min_d = expect_d - 10
max_d = expect_d + 10

min_svp = expect_svp - 10
max_svp = expect_svp + 10

gap = 3
print("gap: gap")
bkz_factor_exp = 1



# min_svp = 140
# max_svp = 170

# min_d = 200
# max_d = 300

# gap = 3
# bkz_factor_exp = 1

# tar_dim = 900
###########################################################

class Scheme:
    def __init__(self, n, m, q, bound, norm_type):
        """
        初始化 Scheme 类的实例。

        :param n: 某个相关的维度或数量
        :param m: 另一个相关的维度或数量
        :param q: 一个模数或其他有意义的数值
        :param bound: 一个边界值
        """
        self.n = n
        self.m = m
        self.q = q
        self.bound = bound
        self.norm = norm_type

    def display_info(self):
        """
        显示该 Scheme 实例的变量信息。
        """
        print(f"n: {self.n}, m: {self.m}, q: {self.q}, bound: {self.bound}")




############ matrix basics ############

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
    print("full matrix with dim %d is loaded, q = %d" % (tar_dim, q))

    return full, q

# return res
def matrix_shrink(full, shrink_to_dim):
    d = len(full)
    res = []
    for row in range(d - shrink_to_dim, d):
        assert(sum([c*c for c in full[row][:d - shrink_to_dim]]) == 0)
        res = [full[row][d - shrink_to_dim:]] + res
    print("matrix has been shrinked from %d to %d" % (len(full), shrink_to_dim))
    return res

# return the squared values of the gso basis.
def gso_init(shrink_mat):

    FPLLL.set_random_seed(1337)
    mat = IntegerMatrix.from_matrix(shrink_mat)
    print("converted matrix to IntegerMatrix")

    A = LLL.reduction(mat)
    M = GSO.Mat(A)
    M.update_gso()
    print("GSO basis is computed")

    return [M.get_r(i, i) for i in range(len(shrink_mat))]

def construct_GSO(n, m, q):
    L = [q*q] * n + [1*1] * (m-n)
    return L


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
    # print(target)
    # print(gaussian_heuristic(r2list[:last_svp]))
    # print(r2list)
    return target / gaussian_heuristic(r2list[:last_svp])

def dim4free_n_div_logn(n):
    if n < 50:
        return 0
    dim4free = int(n / math.log(n))
    return int(min((n - 40) / 2, dim4free))

def complexity_svp1(d):
    max_up = d - (dim4free_n_div_logn(d) - 4)
    global expo
    c = 0
    for dd in range(max_up + 1):
        c += 2**(expo * dd)
    return c

def complexity_svp2(n): # ver2.1
    # CS改
    max_up = n - (n / math.log(n)) + 5
    global expo
    c = 0
    for ii in range(int(max_up)):        
        c += 2**(expo * (max_up - ii))
    return c    

def complexity_svp(n):
    max_up = n - (n / math.log(n)) + 5
    # print("up: ", max_up)
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

def complexity_sieve(n):
    max_up = n
    # print("up: ", max_up)
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
    

# def complexity_svp(n): # ver2.1
#     max_up = n - (n / math.log(n))
#     global expo
#     c = 0
#     for ii in range(int(max_up)):        
#         c += 2**(expo * (max_up - ii))
#     return c    

def complexity_bkz(d, bz):
    return max(1, d - bz + 1) * complexity_svp(bz)

def complexity_overall(d, bzs, last_sieve, bkz_factor=1.0):
    cost = complexity_sieve(last_sieve)
    for bz in bzs:
        cost += complexity_bkz(d, bz) * bkz_factor
    return cost


def gaussian_cdf(mu, sigma, t):
    """
    Compute the cdf of a continuous gaussian random variable with mean mu and standard deviation
    sigma (i.e. computes Pr(X <= t), where X is a gaussian random variable).

    :params mu: the mean of the gaussian random variable.
    :params sigma: the standard deviation of the gaussian random variable.
    :params t: the limit at which to calculate the cdf.

    :returns: the evaluation of the cdf at t.
    """
    return RR((1/2)*(1 + erf((t - mu)/(sqrt(2)*sigma))))

def success_probability(lll_r, q, d, sieve_dim, length_bound):

    if abs(sqrt(lll_r[0]) - q) < 1e-8:  # q-vectors exist
        idx_start = next(i for i, r_ in enumerate(lll_r) if r_ < lll_r[0])
    else:
        idx_start = 0

    if abs(lll_r[-1] - 1) < 1e-8:  # 1-vectors exist
        # Find first 1 length graham-schmidt vector in r (Zone III)
        idx_end = next((i - 1 for i, r_ in enumerate(lll_r) if math.sqrt(r_) <= 1 + 1e-8), d - 1)

    else:
        idx_end = d - 1
    # print(f"idx_start: {idx_start}, idx_end: {idx_end}. ")
    vector_length = sqrt(lll_r[idx_start])
    # print(f"vector_length: {vector_length}")
    gaussian_coords = max(idx_end - idx_start + 1, sieve_dim)
    sigma = vector_length / sqrt(gaussian_coords)

    log_trial_prob = RR(
        log(1 - 2 * gaussian_cdf(0, sigma, -length_bound), 2) * (gaussian_coords)
    )
    log_trial_prob += RR(log((2 * length_bound + 1) / q, 2) * (idx_start))

    N = floor(2 ** (0.2075 * sieve_dim))
    probability = 2 ** min(
    0, log_trial_prob + RR(log(N, 2))
    ) 
    # print(f"probability: {probability}")

    return probability

############ simulator ############

# r is squared values of gso
def sim_bkz(r, block_size): # ok2.1b
    # print("simulate bkz %d" % block_size)
    # bsw18 = BSW18.simulate_bear(r, BKZ.Param(block_size=block_size, max_loops=1))
    # return bsw18[0]
    cn11 = CN11_simulate(r, BKZ.Param(block_size=block_size, max_loops=1))
    return cn11[0]


# return overall cost
def apply_bkz(ori_r, bzs, last_sieve, scheme, verbose=True):
    if verbose:
        print("applying: bzs =", bzs, ", last_sieve =", last_sieve)
    cur_r = copy.copy(ori_r)
    rs = []
    i = 0
    for bz in bzs:
        lastr0 = cur_r[0]
        cur_r = sim_bkz(cur_r, bz)
        # assert(cur_r[0] > lastr0 * 0.999)
        rs += [copy.copy(cur_r)]
        i += 1
        if verbose:
            print("%d: bkz %d" % (i, bz))
    cost = complexity_overall(len(ori_r), bzs, last_sieve)
    if verbose:
        print("done, overall cost ", cost)
    if success_probability(cur_r, scheme.q, len(cur_r), last_sieve, scheme.bound) < 0.99:
        if verbose:
            print("fail!!!!!\n\n")
        return -1, rs
    else:
        if verbose:
            print("success~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
        return cost, rs


#################################### my work ####################################


def reduced_bz_num(ori_r, bz):
    cost = 0
    cur_r = copy.copy(ori_r)
    # print(cur_r[0])
    i = 0
    idx_start = -1

    while True:
        last_r = copy.copy(cur_r)
        cur_r = sim_bkz(cur_r, bz)
        last_idx_start = idx_start

        i += 1
        cost += complexity_bkz(len(ori_r), bz)
        # print(cur_r[0])
        if abs(sqrt(cur_r[0]) - q) < 1e-8:  # q-vectors exist
            idx_start = next(i for i, r_ in enumerate(cur_r) if r_ < cur_r[0])
        else:
            idx_start = 0
        print(f"idx_start: {idx_start}")
        # if (abs(last_r0 - cur_r[0]) < 1e-8):
        if (last_idx_start == idx_start):
            # print(r0rate, 1.001 * last_r0rate, r0rate < 1.001 * last_r0rate)
            # print("iter: ", i)
            break
    return i, cost



# return the number of bz to succeed
def succ_bz_num(ori_r, bz, sieve_dim, scheme):

    last_r0 = 0
    cur_r = copy.copy(ori_r)
    idx_start = -1
    i = 0
    iter_0 = 0
    while success_probability(cur_r, scheme.q, len(cur_r), sieve_dim, scheme.bound) < 0.99:
    # while r0rate < 1.05**2:

        last_r = cur_r
        last_idx_start = idx_start
        cur_r = sim_bkz(cur_r, bz)

        # print(cur_r[0], lastr0 * 0.999, cur_r[0] > lastr0 * 0.999, bz, last_sieve)
        # assert(cur_r[0] > lastr0 * 0.999)
        i += 1
        if abs(sqrt(cur_r[0]) - q) < 1e-8:  # q-vectors exist
            idx_start = next(i for i, r_ in enumerate(cur_r) if r_ < cur_r[0])
        else:
            idx_start = 0
        # if (abs(last_r0 - cur_r[0]) < 1e-8):
        if idx_start == 0:
            iter_0 += 1
        if (last_idx_start == idx_start) and (idx_start > 0):
        # if (last_idx_start == idx_start):
            # print(r0rate, 1.001 * last_r0rate, r0rate < 1.001 * last_r0rate)
            # print("iter: ", i)
            return -1
        if (iter_0 > 5):
            # print("iter: ", i)
            return -1
        last_r0 = cur_r[0]

    return i

def best_bkz_only(ori_r, scheme):
    step_sizes = [64, 16, 4, 1]
    rng = [(max_svp) % 64 + 64, (max_svp)]

    # find the minimal success one
    for step in step_sizes:
        bz = rng[0]
        n_bz = succ_bz_num(ori_r, bz, bz, scheme)
        while bz <= rng[-1] and n_bz < 0:
            bz += step
            if bz <= rng[-1]:
                n_bz = succ_bz_num(ori_r, bz, bz, scheme)
            else:
                n_bz = -1

        if bz > rng[-1]: # cannot success
            return -1, [], -1
        # bz succeed

        rng = [bz - step, bz] # print("find new range", rng)


    cost = complexity_bkz(len(ori_r), bz) * (n_bz) + complexity_sieve(bz)
    bzs = n_bz + 1
    # print(bzs)
    print("bz = %d, num = %d, cost = %.0f" % (bz, bzs, cost))
    # print("last svp = %d, num = %d, cost = %.0f, svp cost = %.0f" % (last_sieve, len(bzs), cost, complexity_svp(last_sieve)))

    return bz, bzs, cost    


def best_succ_bz(ori_r, last_sieve, scheme):
    step_sizes = [64, 16, 4, 1]
    # 这里貌似应该把跨度弄得大一点
    rng = [(last_sieve) % 64 + 64, (last_sieve)]
    # rng = [(last_sieve+20) % 64 + 64, (last_sieve+20)]



    # find the minimal success one
    for step in step_sizes:
        bz = rng[0]
        # print(f"bz: {bz}")
        n_bz = succ_bz_num(ori_r, bz, last_sieve, scheme)
        while bz <= rng[-1] and n_bz < 0:
            bz += step
            if bz <= rng[-1]:
                # print(f"bz: {bz}")
                n_bz = succ_bz_num(ori_r, bz, last_sieve, scheme)
            else:
                n_bz = -1

        if bz > rng[-1]: # cannot success
            return [], -1
        # bz succeed

        rng = [bz - step, bz] # print("find new range", rng)

    pair = []
    global bkz_factor_exp
    for ttbz in range(bz, min(bz + 10, last_sieve+ 1)):
        n_ttbz = succ_bz_num(ori_r, ttbz, last_sieve, scheme)
        print(n_ttbz)
        bzs = [ttbz] * n_ttbz
        pair += [(complexity_overall(len(ori_r), bzs, last_sieve, bkz_factor=bkz_factor_exp), bzs)]
        # print(pair[-1])

    pair.sort()
    print(pair)
    cost = pair[0][0]
    bzs = pair[0][-1]
    print(bzs)
    print("last sieve = %d, bz = %d, num = %d, cost = %.0f, sieve cost = %.0f" % (last_sieve, bzs[0], len(bzs), cost, complexity_sieve(last_sieve)))
    
    # print("last svp = %d, num = %d, cost = %.0f, svp cost = %.0f" % (last_sieve, len(bzs), cost, complexity_svp(last_sieve)))

    return bzs, cost



# return result like (7083793150306219.0, [148, 148, 148], 176)
def get_best_lastsvp(ori_r, scheme):
    res = []
    global max_svp, min_svp
    print("\ntest last_sieve in range: %d ~ %d" % (min_svp, max_svp))
    for last_sieve in range(min_svp, max_svp+1):
        bzs, cost = best_succ_bz(ori_r, last_sieve, scheme)
        if len(bzs) > 0:
            res += [(cost, bzs, last_sieve)]

            if complexity_sieve(last_sieve) > 1.5 * cost:
                break

    if len(res) == 0:
        print("the target cannot be solved by shrinking to %d\n\n" % (len(ori_r)))
        return (-1, [], -1)

    res.sort()
    print("best", res[0], "\n")

    return res[0]

def get_best_d(scheme):
    # full, q =  load_whole_matrix(tar_dim)
    target = scheme.bound * scheme.bound
    best_res = None
    best_d = -1

    for d in range(min_d, max_d, 5):
        print("test shrink dim %d" % d)
        # mat_shrink = matrix_shrink(full, d)
        # lll_r = gso_init(mat_shrink)
        if(d > scheme.m):
            continue
        lll_r = construct_GSO(scheme.n, d, scheme.q)

        res = get_best_lastsvp(lll_r, scheme)

        if res[0] > 0:
            if best_d == -1 or best_res[0] > res[0]:
                best_d = d
                best_res = res

        if best_d > 0 and d >= best_d + 10:
            break


    if best_d == -1:
        print(f"cannot find best d in range [{min_d}, {max_d}]")
        exit(1)
    else:
        print("find rough best d at %d, with result %s" % (best_d, str(best_res)))

    # for d in range(best_d - 5 + 1, min(best_d + 5, max_d)):

    #     print("test shrink dim %d" % d)
    #     # mat_shrink = matrix_shrink(full, d)
    #     # lll_r = gso_init(mat_shrink)
    #     lll_r = construct_GSO(scheme.n, d, scheme.q)

    #     res = get_best_lastsvp(lll_r, scheme)

    #     if res[0] > 0:
    #         if best_res[0] > res[0]:
    #             best_d = d
    #             best_res = res

    print("find final best d at %d, with result (factored cost) %s" % (best_d, str(best_res)))

    return best_d, best_res, q


def success_probability_suffix(ori_r, suffix, last_sieve, scheme):
    cur_r = copy.copy(ori_r)


    for bz in suffix:
        cur_r = sim_bkz(cur_r, bz)


    # print(f"success_probability_suffix: ", remaining_proba)
    return success_probability(cur_r, scheme.q, len(cur_r), last_sieve, scheme.bound)

# return the number of bz to succeed
def succ_bz_num_suffix(ori_r, bz, suffix, last_sieve, scheme):
    max_times = 40
    i = 0

    idx_start = -1
    iter_0 = 0

    cur_r = copy.copy(ori_r)
    while(i < max_times):

        last_idx_start = idx_start

        cur_r = sim_bkz(cur_r, bz)
        i += 1

        if success_probability_suffix(cur_r, suffix, last_sieve, scheme) >= 0.99:
            cost = complexity_overall(len(ori_r), i*[bz] + suffix, last_sieve)
            return (cost, i)
        
        if abs(sqrt(cur_r[0]) - q) < 1e-8:  # q-vectors exist
            idx_start = next(i for i, r_ in enumerate(cur_r) if r_ < cur_r[0])
        else:
            idx_start = 0

        if idx_start == 0:
            iter_0 += 1
        if (last_idx_start == idx_start) and (idx_start > 0):
            # print("iter: ", i)
            return (-1, -1)
        if (iter_0 > 5):
            # print("iter: ", i)
            return (-1, -1)

    return (-1, -1)


# best (cost, bz, n_bz) with + suffix + last_sieve
def best_succ_bz_suffix(ori_r, suffix, last_sieve, scheme):
    assert(len(suffix) > 0)

    ggg = 5
    local_strategy = []
    for bz in range(suffix[0] - ggg, suffix[0]):
        # print(f"bz: {bz}.")
        (cost, n_bz) = succ_bz_num_suffix(ori_r, bz, suffix, last_sieve, scheme)
        if cost > 0:
            return (cost, bz, n_bz) #1111

    if len(local_strategy) == 0:
        return (-1, -1, -1)


def get_best_bzs(scheme, d, cost_bound, init_bzs, last_sieve):    

    # r_file_name = "problem%d_shrink%d" % (tar_dim, d)
    # target = q * q
    # if os.path.exists(r_file_name):
    #     f = open(r_file_name, "r")
    #     all_lines = f.readlines()
    #     f.close()
    #     lll_r = [float(line.replace("\n", "")) for line in all_lines]
    # else:
    #     full, q =  load_whole_matrix(tar_dim)
    #     mat_shrink = matrix_shrink(full, d)
    #     lll_r = gso_init(mat_shrink)

    #     f = open(r_file_name, "w")
    #     for i in range(len(lll_r)):
    #         f.write(str(lll_r[i]) + "\n")
    #     f.close()
    assert(d <= scheme.m)
    lll_r = construct_GSO(scheme.n, d, scheme.q)
    ####
    global gap
    print("let's start with (factored cost) ", cost_bound, init_bzs, last_sieve)
    all_strategy = [(cost_bound, init_bzs, last_sieve, d)] # (cost, bzs, last_sieve, d) # cost_bound is factored, theoretical
    # suffixes = [[bz] for bz in range(init_bzs[0] + gap + 1, init_bzs[0] - 2, -1)]
    suffixes = [[init_bzs[0]]] + [[bz] for bz in range(init_bzs[0] + gap, init_bzs[0], -1)] + [[init_bzs[0] - 1]]

    r = 0
    best_real_cost = cost_bound
    while len(suffixes) > 0:
        print("======== round %d ========\ntodo suffixes:" % (r), suffixes)
        r += 1
        cur_suffix = suffixes[0]
        suffixes = suffixes[1:]
        print("cur_suffix: ", cur_suffix)

        if best_real_cost > 0 and complexity_overall(d, cur_suffix, last_sieve) >= best_real_cost:
            print(complexity_sieve(last_sieve))
            print(complexity_overall(d, cur_suffix, last_sieve))
            print("the suffix is too costly, continue\n\n")
            continue

        (cost, bz, n_bz) = best_succ_bz_suffix(lll_r, cur_suffix, last_sieve, scheme)

        print("fisrt choice:", (cost, bz, n_bz))
        print("baseline: ", (cost_bound, init_bzs, last_sieve, d))

        if cost < 0:
            print("the strategy is infeasible, continue\n\n")
            continue

        if bz < 60 or bz < cur_suffix[-1] - 60:
            all_strategy += [(cost, [bz]*n_bz + cur_suffix, last_sieve, d)]
        else:
            for newbz in range(bz, min(bz + gap, cur_suffix[0]-1) + 1):
                suffixes = [[newbz] + cur_suffix] + suffixes

        all_strategy.sort()
        all_strategy = all_strategy[:5]
        print("best 5 choices:")
        for strat in all_strategy:
            print(strat)
        if len(all_strategy) > 0:
            best_real_cost = all_strategy[0][0]

        print("end round with %d suffixes\n\n" % (len(suffixes)))

    all_strategy.sort()
    if len(all_strategy) > 0:
        print("-------our strategy.-------")
        print(all_strategy[0])
        print(float(log(all_strategy[0][0], 2)))
        best_S = all_strategy[0]
        # best_S = (583769674076637.2, [114, 114, 114, 114, 114, 123, 130, 133], 165, 294)

        # cost = 0
        # for bz in best_S[1]:
        #     cost += complexity_bkz(best_S[-1], bz) * 1
        # print(float(log(cost, 2)))

        print("-------BKZ+SVP.-----------")
        lll_r = construct_GSO(scheme.n, scheme.m, scheme.q)
        bzs = succ_bz_num(lll_r, best_S[1][-1], best_S[2], scheme)
        cost = complexity_overall(best_S[3], [best_S[1][-1]] * bzs, best_S[2]) 
        print(bzs, " * ", best_S[1][-1])
        print(float(log(cost, 2)))



        print("------BKZ-only.----------")
        lll_r = construct_GSO(scheme.n, scheme.m, scheme.q)
        bz, bzs, cost = best_bkz_only(lll_r, scheme)
        print(bzs, " * ", bz)
        print(float(log(cost, 2)))



def check(tar_dim, d, last_sieve, bzs):
    full, q =  load_whole_matrix(tar_dim)
    target = q * q



    mat_shrink = matrix_shrink(full, d)
    lll_r = gso_init(mat_shrink)

    apply_bkz(lll_r, bzs, last_sieve, target, verbose=True)
    return

# check(tar_dim, 213, 114, [59, 64, 69, 74, 79, 84, 89, 92, 95])
# check(tar_dim, 213, 114, [95, 95, 95, 95])
# exit()


# Falcon 512
n=512
q=12289
length_bound=5833.9072
m=1024


# Falcon 1024

n=1024
q=12289
length_bound=8382.4081
m=2048


# lll_r = construct_GSO(n, m, q)
# print("------BKZ-only.----------")
# bz, bzs, cost = best_bkz_only(lll_r, length_bound*length_bound)
# print(bzs, " * ", bz)
# print(float(log(cost, 2)))



# Dilithium2
n=256*4
q=8380417
length_bound=350209
m=256*9

Dilithium2 = Scheme(n, m, q, length_bound, "inf")

# print("------BKZ-only.----------")
# lll_r = construct_GSO(Dilithium2.n, Dilithium2.m, Dilithium2.q)
# bz, bzs, cost = best_bkz_only(lll_r, Dilithium2)
# print(bzs, " * ", bz)
# print(float(log(cost, 2)))
# exit()
# lll_r = construct_GSO(n, m, q)

# print(succ_bz_num(lll_r, 429, 466, length_bound*length_bound))

# apply_bkz(lll_r, [429]*8, 466, length_bound * length_bound, verbose=True)

# print(succ_bz_num_suffix(lll_r, 428, [429], 466, length_bound*length_bound))
# exit()

# print(complexity_overall(2300, [440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 441, 443, 444, 445, 446, 448, 449, 450, 451], 452))
# print(complexity_overall(2300, [452]* 19, 452))
# print(float(log(complexity_bkz(2304, 453), 2)))
# print(complexity_bkz(2304, 453))
# exit()

# best_d, best_res, q = get_best_d(Dilithium2)

# best_d, best_res = 340, (6.928592203637094e+16, [159, 159, 159], 187) 

# get_best_bzs(Dilithium2, 2300, 3.911259692243612e+47, [448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448], 453)
# exit(1)
# get_best_bzs(Dilithium2, best_d, best_res[0], best_res[1], best_res[2])
# exit(1)


# n=256*6
# q=8380417
# length_bound=724481
# m=256*6*2
# Dilithium3 = Scheme(n, m, q, length_bound, "inf")

# best_d, best_res, q = get_best_d(Dilithium3)
# get_best_bzs(Dilithium3, best_d, best_res[0], best_res[1], best_res[2])

# exit(1)

n=256*8
q=8380417
length_bound=769537
m=256*8*2
Dilithium5 = Scheme(n, m, q, length_bound, "inf")

best_d, best_res, q = get_best_d(Dilithium5)
get_best_bzs(Dilithium5, best_d, best_res[0], best_res[1], best_res[2])












