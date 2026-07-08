import json
import os
import glob
import subprocess
import sys

import pytest

VECTORS_DIR = os.path.join(os.path.dirname(__file__), "..", "test_vectors")


def _ensure_vectors():
    """Generate test vectors if they don't exist."""
    if not os.path.isdir(os.path.join(VECTORS_DIR, "univariate")):
        gen_script = os.path.join(os.path.dirname(__file__), "..", "generate_test_vectors.py")
        subprocess.check_call(
            [sys.executable, gen_script, "--outdir", VECTORS_DIR, "--n", "10000"],
            cwd=os.path.dirname(gen_script),
        )


def _load_vectors(subdir):
    _ensure_vectors()
    pattern = os.path.join(VECTORS_DIR, subdir, "*.json")
    vectors = []
    for fpath in sorted(glob.glob(pattern)):
        with open(fpath) as f:
            v = json.load(f)
        vectors.append(v)
    return vectors


def _vector_id(v):
    return v["label"]


def pytest_collect_file(parent, file_path):
    pass


@pytest.fixture(scope="session")
def univariate_vectors():
    return _load_vectors("univariate")


@pytest.fixture(scope="session")
def multivariate_vectors():
    return _load_vectors("multivariate")


def pytest_generate_tests(metafunc):
    if "good_univariate_vector" in metafunc.fixturenames:
        vecs = [v for v in _load_vectors("univariate") if v["tier"] == "good"]
        metafunc.parametrize("good_univariate_vector", vecs, ids=[_vector_id(v) for v in vecs])

    if "bad_univariate_vector" in metafunc.fixturenames:
        vecs = [v for v in _load_vectors("univariate") if v["tier"] == "bad"]
        metafunc.parametrize("bad_univariate_vector", vecs, ids=[_vector_id(v) for v in vecs])

    if "mediocre_univariate_vector" in metafunc.fixturenames:
        vecs = [v for v in _load_vectors("univariate") if v["tier"] == "mediocre"]
        metafunc.parametrize("mediocre_univariate_vector", vecs, ids=[_vector_id(v) for v in vecs])

    if "good_multivariate_vector" in metafunc.fixturenames:
        vecs = [v for v in _load_vectors("multivariate") if v["tier"] == "good"]
        metafunc.parametrize("good_multivariate_vector", vecs, ids=[_vector_id(v) for v in vecs])

    if "bad_multivariate_vector" in metafunc.fixturenames:
        vecs = [v for v in _load_vectors("multivariate") if v["tier"] == "bad"]
        metafunc.parametrize("bad_multivariate_vector", vecs, ids=[_vector_id(v) for v in vecs])
