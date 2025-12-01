# -*- coding: utf-8 -*-
from sage.all import log, exp
from sage.all import line, save, load, identity_matrix, matrix
from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ
from BKZ_Simulator import simulate as CN11_simulate
from BKZ_Simulator import simulate_prob as BSW18_simulate
# from fpylll.tools.bkz_simulator import simulate as CN11_simulate
# import BSW18

import math, copy, os

import time

###########################################################

version = "ver. 3.1, automatic"

# fitted  values
expo1 = .202
expo2 = .249
expo3 = .296


expo = .349
gap = 3
bkz_factor_exp = 1

import logging


def setup_logging(tar_dim):
    # 日志格式（时间-日志级别-消息）
    # log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    log_format = "%(message)s"


    # 配置文件处理器（追加模式，utf-8 编码）
    log_name = f"log_Strategy/log_{tar_dim}3"
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
    logging.info("full matrix with dim %d is loaded, q = %d" % (tar_dim, q))

    return full, q

# return res
def matrix_shrink(full, shrink_to_dim):
    d = len(full)
    res = []
    for row in range(d - shrink_to_dim, d):
        assert(sum([c*c for c in full[row][:d - shrink_to_dim]]) == 0)
        res = [full[row][d - shrink_to_dim:]] + res
    logging.info("matrix has been shrinked from %d to %d" % (len(full), shrink_to_dim))
    return res

# return the squared values of the gso basis.
def gso_init(shrink_mat):

    FPLLL.set_random_seed(1337)
    mat = IntegerMatrix.from_matrix(shrink_mat)
    logging.info("converted matrix to IntegerMatrix")

    A = LLL.reduction(mat)
    M = GSO.Mat(A)
    M.update_gso()
    logging.info("GSO basis is computed")

    return [M.get_r(i, i) for i in range(len(shrink_mat))]



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

def dim4free_vary(rr):
    d = len(rr)
    GH = gaussian_heuristic[rr]
    for f in range(d):
        GH_f = gaussian_heuristic[rr[f:]]
        if math.sqrt(float(d-f)/d) * GH > (math.sqrt(4./3) * GH_f):
            return f-1
    return f

  


def complexity_sieve(n):
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
def sim_bkz(r, block_size, last_svp): # ok2.1b
    # logging.info("simulate bkz %d" % block_size)
    # bsw18 = BSW18.simulate_eta(r, BKZ.Param(block_size=block_size, max_loops=1))
    # return bsw18[0]
    cn11 = CN11_simulate(r, BKZ.Param(block_size=block_size, max_loops=1))
    return cn11[0]


# return overall cost
def apply_bkz(ori_r, bzs, last_svp, target, verbose=True):
    r0rate = get_targh_svp(ori_r, last_svp, target)
    if verbose:
        logging.info(f"applying: bzs = {bzs}, last_svp = {last_svp} ")
        logging.info(f"initial r0rate: {r0rate}")
    cur_r = copy.copy(ori_r)
    rs = []
    i = 0
    for bz in bzs:
        lastr0 = cur_r[0]
        cur_r = sim_bkz(cur_r, bz, last_svp)
        # assert(cur_r[0] > lastr0 * 0.999)
        rs += [copy.copy(cur_r)]
        r0rate = get_targh_svp(cur_r, last_svp, target)
        i += 1
        if verbose:
            logging.info(f"{i}: bkz {bz}, r0rate: {r0rate}")
    cost = complexity_overall(len(ori_r), bzs, last_svp)
    if verbose:
        logging.info(f"done, overall cost {cost} {float(log(cost, 2))}")  # 修正格式
    if r0rate < 1.05**2:
        if verbose:
            logging.info("fail!!!!!\n\n")
        return -1, rs
    else:
        if verbose:
            logging.info("success~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
        return cost, rs


#################################### my work ####################################
def Is_Same(a_r, b_r):
    assert(len(a_r) == len(b_r))
    for i in range(len(a_r)):
        if abs(a_r[i] - b_r[i]) > 1e-8:
            return False
    return True

def reduced_bz_num(ori_r, bz):
    cost = 0
    cur_r = copy.copy(ori_r)
    # logging.info(cur_r[0])  # 单参数无需修改
    i = 0
    
    while True:
        last_r = copy.copy(cur_r)
        cur_r = sim_bkz(cur_r, bz, 0)
        i += 1
        cost += complexity_bkz(len(ori_r), bz)
        # logging.info(cur_r[0])  # 单参数无需修改
        if Is_Same(cur_r, last_r):
            break

    return i, cost



# return the number of bz to succeed
def succ_bz_num(ori_r, bz, last_svp, target):
    r0rate = get_targh_svp(ori_r, last_svp, target)
    # logging.info(r0rate)  # 单参数无需修改
    last_r0rate = 0
    cur_r = copy.copy(ori_r)
    i = 0
    while r0rate < 1.05**2:
        # logging.info(f"i: {i}, r0rate: {r0rate}")  # 修正注释中的格式
        lastr0 = cur_r[0]
        cur_r = sim_bkz(cur_r, bz, last_svp)
        # logging.info(f"cur_r[0]: {cur_r[0]}, lastr0 * 0.999: {lastr0 * 0.999}, comparison: {cur_r[0] > lastr0 * 0.999}, bz: {bz}, last_svp: {last_svp}")  # 修正注释中的格式
        # assert(cur_r[0] > lastr0 * 0.999)
        last_r0rate = r0rate
        r0rate = get_targh_svp(cur_r, last_svp, target)
        i += 1
        if r0rate < 1.001 * last_r0rate:
            # logging.info(f"r0rate: {r0rate}, 1.001 * last_r0rate: {1.001 * last_r0rate}, comparison: {r0rate < 1.001 * last_r0rate}")  # 修正注释中的格式
            return -1
    return i

def best_bkz_only(ori_r, target, max_svp):
    step_sizes = [64, 16, 4, 1]
    rng = [(max_svp - 5) % 64 + 64, (max_svp - 5)]

    # find the minimal success one
    for step in step_sizes:
        bz = rng[0]
        n_bz = succ_bz_num(ori_r, bz, bz, target)
        while bz <= rng[-1] and n_bz < 0:
            bz += step
            if bz <= rng[-1]:
                n_bz = succ_bz_num(ori_r, bz, bz, target)
            else:
                n_bz = -1

        if bz > rng[-1]: # cannot success
            logging.info("not success")
            return [], -1
        # bz succeed

        rng = [bz - step, bz]  # logging.info(f"find new range {rng}")  # 修正注释中的格式


    cost = complexity_bkz(len(ori_r), bz) * (n_bz+1)
    bzs = n_bz + 1
    # logging.info(bzs)  # 单参数无需修改
    logging.info("bz = %d, num = %d, cost = %.0f" % (bz, bzs, cost))
    # logging.info("last svp = %d, num = %d, cost = %.0f, svp cost = %.0f" % (last_svp, len(bzs), cost, complexity_svp(last_svp)))

    return bz, bzs, cost    


def best_succ_bz(ori_r, last_svp, target):
    step_sizes = [64, 16, 4, 1]
    rng = [(last_svp - 5) % 64 + 64, (last_svp - 5)]

    # find the minimal success one
    for step in step_sizes:
        bz = rng[0]
        n_bz = succ_bz_num(ori_r, bz, last_svp, target)
        while bz <= rng[-1] and n_bz < 0:
            bz += step
            if bz <= rng[-1]:
                n_bz = succ_bz_num(ori_r, bz, last_svp, target)
            else:
                n_bz = -1

        if bz > rng[-1]: # cannot success
            return [], -1
        # bz succeed

        rng = [bz - step, bz]  # logging.info(f"find new range {rng}")  # 修正注释中的格式

    pair = []
    global bkz_factor_exp
    for ttbz in range(bz, min(bz + 10, last_svp - 5 + 1)):
        n_ttbz = succ_bz_num(ori_r, ttbz, last_svp, target)
        bzs = [ttbz] * n_ttbz
        pair += [(complexity_overall(len(ori_r), bzs, last_svp, bkz_factor=bkz_factor_exp), bzs)]
        # logging.info(pair[-1])  # 单参数无需修改

    pair.sort()
    cost = pair[0][0]
    bzs = pair[0][-1]
    # logging.info(bzs)  # 单参数无需修改
    logging.info("last svp = %d, bz = %d, num = %d, cost = %.0f, svp cost = %.0f" % (last_svp, bzs[0], len(bzs), cost, complexity_sieve(last_svp)))
    # logging.info("last svp = %d, num = %d, cost = %.0f, svp cost = %.0f" % (last_svp, len(bzs), cost, complexity_svp(last_svp)))

    return bzs, cost



# return result like (7083793150306219.0, [148, 148, 148], 176)
def get_best_lastsvp(ori_r, target, min_svp, max_svp):
    res = []
    # global max_svp, min_svp
    logging.info("\ntest last_svp in range: %d ~ %d" % (min_svp, max_svp))
    for last_svp in range(min_svp, max_svp+1):
        bzs, cost = best_succ_bz(ori_r, last_svp, target)
        if len(bzs) > 0:
            res += [(cost, bzs, last_svp)]

            if complexity_sieve(last_svp) > 1.5 * cost:
                break

    if len(res) == 0:
        logging.info("the target cannot be solved by shrinking to %d\n\n" % (len(ori_r)))
        return (-1, [], -1)

    res.sort()
    logging.info(f"best {res[0]}\n")  # 修正格式

    return res[0]

def get_best_d(tar_dim, min_d, max_d, min_svp, max_svp):
    full, q =  load_whole_matrix(tar_dim)
    target = q * q
    best_res = None
    best_d = -1

    for d in range(min_d, max_d, 5):
        logging.info("test shrink dim %d" % d)
        mat_shrink = matrix_shrink(full, d)
        lll_r = gso_init(mat_shrink)

        res = get_best_lastsvp(lll_r, target, min_svp, max_svp)

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
        mat_shrink = matrix_shrink(full, d)
        lll_r = gso_init(mat_shrink)
        res = get_best_lastsvp(lll_r, target, min_svp, max_svp)

        if res[0] > 0:
            if best_res[0] > res[0]:
                best_d = d
                best_res = res

    logging.info("find final best d at %d, with result (factored cost) %s" % (best_d, str(best_res)))

    return best_d, best_res, q




# return the number of bz to succeed
def succ_bz_num_suffix(ori_r, bz, suffix, last_svp, target):
    max_times = 8
    max_cost, _ = apply_bkz(ori_r, [bz] * max_times + suffix, last_svp, target, verbose=False)
    if max_cost < 0:
        return (-1, -1)

    for i in range(1, max_times):
        cost, _ = apply_bkz(ori_r, [bz] * i + suffix, last_svp, target, verbose=False)
        if cost > 0:
            return (cost, i)
    return (max_cost, max_times)


# best (cost, bz, n_bz) with + suffix + last_svp
def best_succ_bz_suffix(ori_r, suffix, last_svp, target):
    assert(len(suffix) > 0)

    ggg = 5
    local_strategy = []
    for bz in range(suffix[0] - ggg, suffix[0]+1):
        (cost, n_bz) = succ_bz_num_suffix(ori_r, bz, suffix, last_svp, target)
        if cost > 0:
            return (cost, bz, n_bz) #1111

    if len(local_strategy) == 0:
        return (-1, -1, -1)


def get_best_bzs(tar_dim, d, cost_bound, init_bzs, last_svp, q, max_svp):    

    r_file_name = "problem%d_shrink%d" % (tar_dim, d)
    target = q * q
    if os.path.exists(r_file_name):
        f = open(r_file_name, "r")
        all_lines = f.readlines()
        f.close()
        lll_r = [float(line.replace("\n", "")) for line in all_lines]
    else:
        full, q =  load_whole_matrix(tar_dim)
        mat_shrink = matrix_shrink(full, d)
        lll_r = gso_init(mat_shrink)

        f = open(r_file_name, "w")
        for i in range(len(lll_r)):
            f.write(str(lll_r[i]) + "\n")
        f.close()

    assert(target > 0)
    ####
    global gap
    logging.info(f"let's start with (factored cost) {cost_bound} {init_bzs} {last_svp}")  # 修正格式
    all_strategy = [(cost_bound, init_bzs, last_svp, d)] # (cost, bzs, last_svp, d) # cost_bound is factored, theoretical
    # suffixes = [[bz] for bz in range(init_bzs[0] + gap + 1, init_bzs[0] - 2, -1)]
    suffixes = [[init_bzs[0]]] + [[bz] for bz in range(init_bzs[0] + gap, init_bzs[0], -1)] + [[init_bzs[0] - 1]]

    r = 0
    best_real_cost = cost_bound
    while len(suffixes) > 0:
        logging.info(f"======== round {r} ========\ntodo suffixes: {suffixes}")  # 修正格式
        r += 1
        cur_suffix = suffixes[0]
        suffixes = suffixes[1:]
        logging.info(f"cur_suffix: {cur_suffix}")  # 修正格式

        if best_real_cost > 0 and complexity_overall(d, cur_suffix, last_svp) >= best_real_cost:
            logging.info("the suffix is too costly, continue\n\n")
            continue

        (cost, bz, n_bz) = best_succ_bz_suffix(lll_r, cur_suffix, last_svp, target)

        logging.info(f"first choice: { (cost, bz, n_bz) }")  # 修正格式（同时修正拼写错误fisrt→first）
        logging.info(f"baseline: { (cost_bound, init_bzs, last_svp, d) }")  # 修正格式

        if cost < 0:
            logging.info("the strategy is infeasible, continue\n\n")
            continue

        if bz <= 70 or bz <= cur_suffix[-1] - 40:
            all_strategy += [(cost, [bz]*n_bz + cur_suffix, last_svp, d)]
        else:
            for newbz in range(bz, min(bz + gap, cur_suffix[0]) + 1):
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
    if len(all_strategy) > 0:
        logging.info("-------our strategy.-------")
        logging.info(all_strategy[0])
        logging.info(float(log(all_strategy[0][0], 2)))
        best_S = all_strategy[0]
        # best_S = (583769674076637.2, [114, 114, 114, 114, 114, 123, 130, 133], 165, 294)

        # cost = 0
        # for bz in best_S[1]:
        #     cost += complexity_bkz(best_S[-1], bz) * 1
        # logging.info(float(log(cost, 2)))

        logging.info("-------BKZ+SVP.-----------")
        bzs = succ_bz_num(lll_r, best_S[1][-1], best_S[2], target)
        cost = complexity_overall(best_S[3], [best_S[1][-1]] * bzs, best_S[2]) 
        logging.info(f"{bzs} * {best_S[1][-1]}")  # 修正格式
        logging.info(float(log(cost, 2)))


        logging.info("------bkz reduced.------")
        bzs, cost = reduced_bz_num(lll_r, best_S[1][-1])
        logging.info(f"{bzs} * {best_S[1][-1]}")  # 修正格式
        logging.info(float(log(cost, 2)))

        logging.info("------BKZ-only.----------")
        bz, bzs, cost = best_bkz_only(lll_r, target, max_svp)
        logging.info(f"{bzs} * {bz}")  # 修正格式
        logging.info(float(log(cost, 2)))



def check(tar_dim, d, last_svp, bzs):
    full, q =  load_whole_matrix(tar_dim)
    target = q * q


    mat_shrink = matrix_shrink(full, d)
    lll_r = gso_init(mat_shrink)

    apply_bkz(lll_r, bzs, last_svp, target, verbose=True)
    return



def SIS_startegy_search(target_dim, expect_d, expext_svp):

    min_svp = expect_svp - 10
    max_svp = expect_svp + 10

    min_d = expect_d - 10
    max_d = expect_d + 10



    start = time.time()

    best_d, best_res, q = get_best_d(tar_dim, min_d, max_d, min_svp, max_svp)

    get_best_bzs(tar_dim, best_d, best_res[0], best_res[1], best_res[2], q, max_svp)

    end = time.time()
    logging.info("total time: %.2f seconds" % (end - start))


if __name__ == "__main__":
    # 经验值
    
    # S = [169]
    # d=340
    # r=178
    # print(float(log(complexity_overall(d, S, r), 2)))

    # tar_dim = 1100
    # setup_logging(tar_dim)
    # full, q =  load_whole_matrix(tar_dim)
    # target = q * q
    # d = 340
    # r  =178
    # S = [167]*5
    # mat_shrink = matrix_shrink(full, d)
    # lll_r = gso_init(mat_shrink)
    # apply_bkz(lll_r, S, r, target, verbose=True)
    # exit(1)

    tar_dim = 1100
    expect_d = 337
    expect_svp = 178

    setup_logging(tar_dim)
    SIS_startegy_search(tar_dim, expect_d, expect_svp)
    exit(1)


    tar_dim = 750
    expect_d = 208
    expect_svp = 104
    setup_logging(tar_dim)
    SIS_startegy_search(tar_dim, expect_d, expect_svp)


    tar_dim = 700
    expect_d = 190
    expect_svp = 90

    setup_logging(tar_dim)
    SIS_startegy_search(tar_dim, expect_d, expect_svp)


    tar_dim = 800
    expect_d = 227
    expect_svp = 117

    setup_logging(tar_dim)
    SIS_startegy_search(tar_dim, expect_d, expect_svp)


    tar_dim = 900
    expect_d = 262
    expect_svp = 141

    setup_logging(tar_dim)
    SIS_startegy_search(tar_dim, expect_d, expect_svp)

    tar_dim = 1000
    expect_d = 294
    expect_svp = 165

    setup_logging(tar_dim)
    SIS_startegy_search(tar_dim, expect_d, expect_svp)

    tar_dim = 1100
    expect_d = 337
    expect_svp = 178

    setup_logging(tar_dim)
    SIS_startegy_search(tar_dim, expect_d, expect_svp)

    tar_dim = 1200
    expect_d = 383
    expect_svp = 213

    setup_logging(tar_dim)
    SIS_startegy_search(tar_dim, expect_d, expect_svp)


