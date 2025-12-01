from sage.all import log, exp
from sage.all import line, save, load, identity_matrix, matrix
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




tar_dim = 800
expect_d = 227
expect_svp = 117

tar_dim = 900
expect_d = 262
expect_svp = 141

tar_dim = 1000
expect_d = 294
expect_svp = 165

tar_dim = 1100
expect_d = 337
expect_svp = 188

tar_dim = 1200
expect_d = 383
expect_svp = 213

min_svp = expect_svp - 10
max_svp = expect_svp + 10
# min_svp = 110
# max_svp = 130

min_d = expect_d - 10
max_d = expect_d + 10
# min_d = 240
# max_d = 270

expect_d = 1024
expect_svp = 466

expect_d = 2048
expect_svp = 1040
min_d = expect_d - 10
max_d = expect_d + 1

min_svp = expect_svp - 20
max_svp = expect_svp + 20

gap = 2
bkz_factor_exp = 1

# min_svp = 140
# max_svp = 170

# min_d = 200
# max_d = 300

# gap = 3
# bkz_factor_exp = 1

# tar_dim = 900
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
    # print("simulate bkz %d" % block_size)
    cn11 = CN11_simulate(r, BKZ.Param(block_size=block_size, max_loops=1))
    return cn11[0]


# return overall cost
def apply_bkz(ori_r, bzs, last_svp, target, verbose=True):
    r0rate = get_targh_svp(ori_r, last_svp, target)
    if verbose:
        print("applying: bzs =", bzs, ", last_svp =", last_svp)
        print("initial r0rate: ", r0rate)
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
            print("%d: bkz %d, r0rate: " % (i, bz), r0rate)
    cost = complexity_overall(len(ori_r), bzs, last_svp)
    if verbose:
        print("done, overall cost ", cost)
    if r0rate < 1.05**2:
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
    
    while True:
        last_r = copy.copy(cur_r)
        cur_r = sim_bkz(cur_r, bz, 0)
        i += 1
        cost += complexity_bkz(len(ori_r), bz)
        # print(cur_r[0])
        if cur_r == last_r:
            break
    return i, cost



# return the number of bz to succeed
def succ_bz_num(ori_r, bz, last_svp, target):
    r0rate = get_targh_svp(ori_r, last_svp, target)
    print(f"{bz}, {last_svp}, ",r0rate)
    last_r0rate = 0
    cur_r = copy.copy(ori_r)
    i = 0
    while r0rate < 1.05**2:
        # print(i, " ", r0rate)
        # lastr0 = cur_r[0]
        cur_r = sim_bkz(cur_r, bz, last_svp)
        # print(cur_r[0], lastr0 * 0.999, cur_r[0] > lastr0 * 0.999, bz, last_svp)
        # assert(cur_r[0] > lastr0 * 0.999)
        last_r0rate = r0rate
        r0rate = get_targh_svp(cur_r, last_svp, target)
        i += 1
        if r0rate < 1.001 * last_r0rate:
            # print(r0rate, 1.001 * last_r0rate, r0rate < 1.001 * last_r0rate)
            print("iter: ", i, " ", r0rate)
            return -1
    print(f"bz: {bz}, succ_bz: {i}")
    return i

def best_bkz_only(ori_r, target):
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
            return [], -1
        # bz succeed

        rng = [bz - step, bz] # print("find new range", rng)


    cost = complexity_bkz(len(ori_r), bz) * (n_bz+1)
    bzs = n_bz + 1
    # print(bzs)
    print("bz = %d, num = %d, cost = %.0f" % (bz, bzs, cost))
    # print("last svp = %d, num = %d, cost = %.0f, svp cost = %.0f" % (last_svp, len(bzs), cost, complexity_svp(last_svp)))

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

        rng = [bz - step, bz] # print("find new range", rng)

    pair = []
    global bkz_factor_exp
    for ttbz in range(bz, min(bz + 10, last_svp - 5 + 1)):
        n_ttbz = succ_bz_num(ori_r, ttbz, last_svp, target)
        # print(n_ttbz)
        bzs = [ttbz] * n_ttbz
        pair += [(complexity_overall(len(ori_r), bzs, last_svp, bkz_factor=bkz_factor_exp), bzs)]
        # print(pair[-1])

    pair.sort()
    cost = pair[0][0]
    bzs = pair[0][-1]
    print(bzs)
    print("last svp = %d, bz = %d, num = %d, cost = %.0f, svp cost = %.0f" % (last_svp, bzs[0], len(bzs), cost, complexity_sieve(last_svp)))
    # print("last svp = %d, num = %d, cost = %.0f, svp cost = %.0f" % (last_svp, len(bzs), cost, complexity_svp(last_svp)))

    return bzs, cost



# return result like (7083793150306219.0, [148, 148, 148], 176)
def get_best_lastsvp(ori_r, target):
    res = []
    global max_svp, min_svp
    print("\ntest last_svp in range: %d ~ %d" % (min_svp, max_svp))
    for last_svp in range(min_svp, max_svp+1):
        bzs, cost = best_succ_bz(ori_r, last_svp, target)
        if len(bzs) > 0:
            res += [(cost, bzs, last_svp)]

            if complexity_sieve(last_svp) > 1.5 * cost:
                break

    if len(res) == 0:
        print("the target cannot be solved by shrinking to %d\n\n" % (len(ori_r)))
        return (-1, [], -1)

    res.sort()
    print("best", res[0], "\n")

    return res[0]

def get_best_d(n, m, q, target):
    # full, q =  load_whole_matrix(tar_dim)
    # target = q * q
    best_res = None
    best_d = -1

    for d in range(min_d, max_d, 5):
        print("test shrink dim %d" % d)
        # mat_shrink = matrix_shrink(full, d)
        # lll_r = gso_init(mat_shrink)
        if(d > m):
            continue
        lll_r = construct_GSO(n, d, q)

        res = get_best_lastsvp(lll_r, target)

        if res[0] > 0:
            if best_d == -1 or best_res[0] > res[0]:
                best_d = d
                best_res = res

        if best_d > 0 and d >= best_d + 10:
            break


    if best_d == -1:
        print("cannot find best d in range [%d, %d)" % (min_d, max_d))
        exit(1)
    else:
        print("find rough best d at %d, with result %s" % (best_d, str(best_res)))

    # for d in range(1024, 1025):
    for d in range(best_d - 5 + 1, min(best_d + 5, max_d)):

        print("test shrink dim %d" % d)
        # mat_shrink = matrix_shrink(full, d)
        # lll_r = gso_init(mat_shrink)
        lll_r = construct_GSO(n, d, q)

        res = get_best_lastsvp(lll_r, target)

        if res[0] > 0:
            if best_res[0] > res[0]:
                best_d = d
                best_res = res

    print("find final best d at %d, with result (factored cost) %s" % (best_d, str(best_res)))

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
    for bz in range(suffix[0] - ggg, suffix[0]):
        (cost, n_bz) = succ_bz_num_suffix(ori_r, bz, suffix, last_svp, target)
        if cost > 0:
            return (cost, bz, n_bz) #1111

    if len(local_strategy) == 0:
        return (-1, -1, -1)


def get_best_bzs(n ,m, q, target, d, cost_bound, init_bzs, last_svp):    

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
    assert(d <= m)
    lll_r = construct_GSO(n, d, q)
    assert(target > 0)
    ####
    global gap
    print("let's start with (factored cost) ", cost_bound, init_bzs, last_svp)
    all_strategy = [(cost_bound, init_bzs, last_svp, d)] # (cost, bzs, last_svp, d) # cost_bound is factored, theoretical
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

        if best_real_cost > 0 and complexity_overall(d, cur_suffix, last_svp) >= best_real_cost:
            print("the suffix is too costly, continue\n\n")
            continue

        (cost, bz, n_bz) = best_succ_bz_suffix(lll_r, cur_suffix, last_svp, target)

        print("fisrt choice:", (cost, bz, n_bz))
        print("baseline: ", (cost_bound, init_bzs, last_svp, d))

        if cost < 0:
            print("the strategy is infeasible, continue\n\n")
            continue

        if bz < 60 or bz < cur_suffix[-1] - 40:
            all_strategy += [(cost, [bz]*n_bz + cur_suffix, last_svp, d)]
        else:
            for newbz in range(bz, min(bz + gap, cur_suffix[0] - 1) + 1):
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

        print("-------BKZ+SVP.----------")
        lll_r = construct_GSO(n, m, q)
        bzs = succ_bz_num(lll_r, best_S[1][-1], best_S[2], target)
        cost = complexity_overall(best_S[3], [best_S[1][-1]] * bzs, best_S[2]) 
        print(f"{bzs} * {best_S[1][-1]} + {best_S[2]}")
        print(float(log(cost, 2)))


        print("------bkz reduced.-------")
        lll_r = construct_GSO(n, m, q)
        bzs, cost = reduced_bz_num(lll_r, best_S[1][-1])
        print(bzs, " * ", best_S[1][-1])
        print(float(log(cost, 2)))

        print("------ BKZ-only.---------")
        lll_r = construct_GSO(n, m, q)
        bz, bzs, cost = best_bkz_only(lll_r, target)
        print(bzs, " * ", bz)
        print(float(log(cost, 2)))



def check(tar_dim, d, last_svp, bzs):
    full, q =  load_whole_matrix(tar_dim)
    target = q * q



    mat_shrink = matrix_shrink(full, d)
    lll_r = gso_init(mat_shrink)

    apply_bkz(lll_r, bzs, last_svp, target, verbose=True)
    return

# check(tar_dim, 213, 114, [59, 64, 69, 74, 79, 84, 89, 92, 95])
# check(tar_dim, 213, 114, [95, 95, 95, 95])
# exit()


# Falcon 512
# n=512
# q=12289
# length_bound=5833.9072
# m=1024


# Falcon 1024
n=1024
q=12289
length_bound=8382.4081
m=2048

# lll_r = construct_GSO(n, m, q)

# print(succ_bz_num(lll_r, 429, 466, length_bound*length_bound))

# apply_bkz(lll_r, [429]*8, 466, length_bound * length_bound, verbose=True)

# print(succ_bz_num_suffix(lll_r, 428, [429], 466, length_bound*length_bound))
# exit()

# print("------BKZ-only.----------")
# bz, bzs, cost = best_bkz_only(lll_r, length_bound*length_bound)
# print(bzs, " * ", bz)
# print(float(log(cost, 2)))
# exit()

best_d, best_res, q = get_best_d(n, m, q, length_bound * length_bound)

# best_d, best_res = 340, (6.928592203637094e+16, [159, 159, 159], 187) 

get_best_bzs(n, m, q, length_bound * length_bound, best_d, best_res[0], best_res[1], best_res[2])

exit(1)


# # print(math.log(complexity_svp(190), 2))
# # print(math.log(complexity_svp(213), 2))
# # exit(1)


# def cost1(tar_dim, d, r, S):

#     full, q =  load_whole_matrix(tar_dim)
#     target = q * q


#     mat_shrink = matrix_shrink(full, d)
#     lll_r = gso_init(mat_shrink)
#     res = apply_bkz(lll_r, S, r, target)[0]
#     print(math.log(res, 2))
#     cost = 0
#     for bz in S:
#         cost = complexity_bkz(d, bz)
#         print(bz, math.log(cost, 2))
#         print()
#     print(math.log(complexity_sieve(r), 2))
#     print()
#     return res 



# # c = cost1(1200, 383, 213, [113, 113, 113, 113, 113, 113, 122, 126, 132, 135, 139, 145, 150, 156, 162, 166, 172, 178, 183])
# # print(c)
# print(math.log(complexity_sieve(180),2 ))
# exit()



# # cost1(900, 262, 141, [8, 16, 32, 64, 128, 128])
# # cost1(900, 262, 141, [68, 68, 68, 68, 68, 68, 72, 76, 79, 80, 85, 88, 93, 96, 102, 106, 110])
# # cost1(900, 264, 139, [104, 104, 104, 104, 104, 104, 113])
# avr = []
# c = cost1(800, 230, 140, [60, 65, 70, 75, 75])
# print(c / 2645)
# avr.append(c / 2645)
# c = cost1(775, 220, 135, [65, 65, 70, 75])
# print(c / 1572)
# avr.append(c / 1572)
# c = cost1(750, 210, 125, [60, 65, 70, 75])
# print(c / 1027)
# avr.append(c / 1027)
# c = cost1(725, 200, 115, [60, 65, 70, 70])
# print(c / 765)
# avr.append(c / 765)
# c = cost1(700, 190, 105, [65, 70, 70])
# print(c / 517)
# avr.append(c / 517)

# print(avr)
# avrage = sum(avr) / 5
# lbound = avrage - avrage / 10
# hbound = avrage + avrage / 10
# print("avrage: %f, the interval: [%f, %f]." % (avrage, lbound, hbound))
# for c in avr:
#     print((c >= lbound and c <= hbound))
# # avr = [333241, 327144, 286246, 301986, 292711]


# exit(1)





# # for bz in range(140, 161, 5):
# #     print(float(log(complexity_bkz(340, bz), 2)))
# # exit(1)
# ############################## main ##############################



# # best_S = (583769674076637.2, [114, 114, 114, 114, 114, 123, 130, 133], 165, 294)
# # full, q =  load_whole_matrix(tar_dim)
# # mat_shrink = matrix_shrink(full, best_S[-1])
# # lll_r = gso_init(mat_shrink)

# # bzs = succ_bz_num(lll_r, 133, 165)
# # cost = bzs * complexity_bkz(294, 133) + complexity_svp(165)
# # print(float(log(cost, 2)))

# # cost = 0
# # for bz in best_S[1]:
# #     cost += complexity_bkz(best_S[-1], bz) * 1
# # print(float(log(cost, 2)))
# # bzs, cost = reduced_bz_num(lll_r, best_S[1][-1])
# # print(bzs, " * ", best_S[1][-1])
# # print(float(log(cost, 2)))

# # exit(1)


# tar_dim = 800

# best_d, best_res, q = get_best_d(tar_dim)

# # best_d, best_res = 340, (6.928592203637094e+16, [159, 159, 159], 187) 

# get_best_bzs(tar_dim, best_d, best_res[0], best_res[1], best_res[2], q)

# exit(1)
# #1200 1.236950505771844e+19, [113, 113, 113, 113, 113, 113, 122, 126, 132, 135, 139, 145, 150, 156, 162, 166, 172, 178, 183], 213, 383





# tar_dim = 1050


# best_d, best_res = get_best_d(tar_dim)

# # find final best d at 318, with result (6750597512358777.0, [147, 147, 147, 147], 175)
# # best_d, best_res = 318, (6877418752660590.0, [147, 147, 147, 147], 175)

# get_best_bzs(tar_dim, best_d, best_res[0], best_res[1], best_res[2])

# exit(1)




