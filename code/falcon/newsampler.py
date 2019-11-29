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
sigmin = 1.2
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


def make_cdt(pdt):
    len_pdt = len(pdt)
    cdt = pdt[:-1]
    for i in range(1, len_pdt - 1):
        cdt[i] += cdt[i - 1]
    return cdt

# Compute the CDT from the PDT
halfgaussian_cdt = make_cdt(halfgaussian_pdt)


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
            return floor(center) + z
