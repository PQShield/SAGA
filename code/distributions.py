"""
Shared discrete-Gaussian table helper.

Single source of truth for turning (mu, sigma) into aligned (support, probs)
numpy arrays. Previously duplicated verbatim as `_pdt_arrays` (calibration.py,
scripts/run_extensive_calibration.py) and `pdt_arrays` (univariate_tests.py).
"""
import numpy as np

from saga import make_gaussian_pdt


def pdt_arrays(mu, sigma, tau=14):
    """
    Return the discrete Gaussian D_{Z, sigma, mu} as aligned numpy arrays
    (support, probs).

    `tau` is accepted for backward-compatible call sites but is not used here:
    the tail cut comes from make_gaussian_pdt, which uses the module-level tau
    in saga.py.
    """
    pdt = make_gaussian_pdt(mu, sigma)
    support = np.array(sorted(pdt.keys()))
    probs = np.array([pdt[z] for z in support])
    return support, probs
