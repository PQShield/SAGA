# Importing dependencies
from random import randint
from math import floor, exp
# For debugging purposes
import sys
if sys.version_info >= (3, 4):
    from importlib import reload  # Python 3.4+ only.

# Upper bound on all the values of sigma
sigma0 = 1.8205
# Lower bound on all the values of sigma
sigmin = 1.3
# Precision of the CDT
cdt_precision = 72

# New probability distribution table from [PRR19]
halfgaussian_pdt = [
    1697680241746640300030,
    1459943456642912959616,
    928488355018011056515,
    436693944817054414619,
    151893140790369201013,
    39071441848292237840,
    7432604049020375675,
    1045641569992574730,
    108788995549429682,
    8370422445201343,
    476288472308334,
    20042553305308,
    623729532807,
    14354889437,
    244322621,
    3075302,
    28626,
    197,
    1]

# When in the same format as in Falcon's reference code, looks like this
# 6031371 13708371 13035518
# 5186761 1487980 12270720
# 3298653 4688887 5511555
# 1551448 9247616 9467675
# 539632 14076116 5909365
# 138809 10836485 13263376
# 26405 15335617 16601723
# 3714 14514117 13240074
# 386 8324059 3276722
# 29 12376792 7821247
# 1 11611789 3398254
# 0 1194629 4532444
# 0 37177 2973575
# 0 855 10369757
# 0 14 9441597
# 0 0 3075302
# 0 0 28626
# 0 0 197
# 0 0 1


def make_cdt(pdt):
    len_pdt = len(pdt)
    cdt = pdt[:-1]
    for i in range(1, len_pdt - 1):
        cdt[i] += cdt[i - 1]
    return cdt

# Compute the CDT from the PDT
halfgaussian_cdt = make_cdt(halfgaussian_pdt)

# When in the same format as in Falcon's reference code, looks like this:
# 6031371 13708371 13035518
# 11218132 15196352 8529022
# 14516786 3108023 14040577
# 16068234 12355640 6731036
# 16607867 9654540 12640401
# 16746677 3713810 9126561
# 16773083 2272212 8951068
# 16776798 9114 5413926
# 16777184 8333173 8690648
# 16777214 3932749 16511895
# 16777215 15544539 3132933
# 16777215 16739168 7665377
# 16777215 16776345 10638952
# 16777215 16777201 4231493
# 16777215 16777215 13673090
# 16777215 16777215 16748392
# 16777215 16777215 16777018
# 16777215 16777215 16777215


#        0,        0,         1,
#        0,        0,       198,
#        0,        0,     28824,
#        0,        0,   3104126,
#        0,       14,  12545723,
#        0,      870,   6138264,
#        0,    38047,   9111839,
#        0,  1232676,  13644283,
#        1, 12844466,    265321,
#       31,  8444042,   8086568,
#      417, 16768101,  11363290,
#     4132, 14505003,   7826148,
#    30538, 13063405,   7650655,
#   169348,  7122675,   4136815,
#   708981,  4421575,  10046180,
#  2260429, 13669192,   2736639,
#  5559083,  1580863,   8248194,
# 10745844,  3068844,   3741698

def sampler0():
    """Sample from a half-Gaussian."""
    r = randint(0, (1 << cdt_precision) - 1)
    z0 = 0
    for elt in halfgaussian_cdt:
        z0 += (r >= elt)
    return z0


# Precision in bits of p in BerExp
berexp_p = 64
# Precision in bits of each rand in BerExp
berexp_rand = 8
rand_mask = (1 << berexp_rand) - 1
# Sanity check
assert(berexp_p % berexp_rand == 0)


def berexp(x, sf):
    """
    Return True with a probability exp(-x).
    sf is a scaling factor.
    """
    # FIXME
    p = int(exp(-x) * sf * (1 << berexp_p)) - 1
    i = berexp_p
    # Careful: in C, i must be unsigned otherwise it might loop forever!
    while(i > 0):
        i -= berexp_rand
        r = randint(0, (1 << berexp_rand) - 1)
        # Si la randomness est plus faible que p, on accepte
        if r < ((p >> i) & rand_mask):
            return True
        # Si la randomness est plus elevee que p, on refuse
        if r > ((p >> i) & rand_mask):
            return False
        # Sinon, on continue jusqu'a ce que i = 0
    return True


def samplerz(center, sigma):
    """
    Sample from a discrete Gaussian with specified center and sigma.
    """
    assert(sigma < sigma0)
    assert(sigma >= sigmin)
    # c0 is the fractional part of center
    c0 = center - floor(center)
    sf = sigma / sigma0
    while(1):
        z0 = sampler0()
        b = randint(0, 1)
        z = ((b << 1) - 1) * z0 + b
        x = ((z - c0) ** 2) / (2 * (sigma ** 2)) - (z0 ** 2) / (2 * (sigma0 ** 2))
        if berexp(x, sf) is True:
            #print(counter)
            return floor(center) + z
