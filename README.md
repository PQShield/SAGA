# Statistically Acceptable GAussians (SAGA)

![Joke-Logo](https://github.com/PQShield/SAGA/blob/master/code/testdata/saga.png)

Authored by James Howe, Thomas Prest, Thomas Ricosset, and Mélissa Rossi - 27-Jan-2020

Code from the paper available at: https://eprint.iacr.org/2019/1411

## Introduction

SAGA (Statistically Acceptable GAussians) is a test suite proposal for verfying statistical correctness for univariate and multivariate Gaussians. The paper accompanying this code has been published at PQCrypto 2020 and is also available on [ePrint](https://eprint.iacr.org/2019/1411). The following will briefly describe how to setup and use the python script.

## Installation

This standalone implementation should be able to run on most machines. We have provided a `requirements.txt` file to install all the dependencies; install these can be done by simple running `pip install -r requirements.txt` for Python 2 or `pip3 install -r requirements.txt` for Python 3.

## How to use

Along with the main file to run these statistical tests, `saga.py`, we also provide code for our proposed sampler [1] in the files `sampler.c`, `sampler.py`, where `sampler_rep.py` is a file we use to get data on the repetition rate. We also provide [falcon/](https://github.com/PQShield/SAGA/tree/master/code/falcon) and [testdata/](https://github.com/PQShield/SAGA/tree/master/code/testdata) for python implementations of [Falcon](https://falcon-sign.info/) and its output values.

Points we need to include:

1. How to add your own .csv etc files.
2. What you can edit/adapt, if you want
3. Breifly describe univariate, multivariate, and supplementary tests.

***

[1] James Howe, Thomas Prest, Thomas Ricosset, and Mélissa Rossi. Isochronous gaussian sampling: From inception to implementation. Cryptology ePrint Archive, Report 2019/1411, 2019. https://eprint.iacr.org/2019/1411.

**Provided with absolutely no warranty whatsoever**
