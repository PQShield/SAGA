"""
Run current SAGA tests against generated test vectors and report a baseline.

Usage:
    cd code/
    python run_baseline.py [--vectors-dir test_vectors]

Output: baseline_results.json with pass/fail for each vector against each test.
"""
import json
import os
import sys
import glob

# SAGA imports (must run from code/ directory)
from saga import UnivariateSamples, pmin


def run_univariate(vector):
    """Run current SAGA univariate tests on a vector. Return results dict."""
    mu = vector["params"]["mu"]
    sigma = vector["params"]["sigma"]
    samples = vector["samples"]

    results = {}
    try:
        uv = UnivariateSamples(mu, sigma, samples)
        results["chi2_pass"] = bool(uv.chi2_pvalue > pmin)
        results["chi2_pvalue"] = float(uv.chi2_pvalue)
        results["chi2_stat"] = float(uv.chi2_stat)
        results["outliers"] = int(uv.outlier)
        results["is_valid"] = bool(uv.is_valid)
        results["empirical_mean"] = float(uv.mean)
        results["empirical_stdev"] = float(uv.stdev)
        results["empirical_skewness"] = float(uv.skewness)
        results["empirical_kurtosis"] = float(uv.kurtosis)
        ext = uv.run_extended_battery(samples=samples, mc_B=200)
        results["extended_pass"] = ext["all_pass"]
        results["is_valid_extended"] = bool(uv.is_valid_extended)
        results["extended_details"] = {
            k: ext[k].get("passes", None) if isinstance(ext[k], dict) else ext[k]
            for k in ext if k != "all_pass"
        }
        results["error"] = None
    except Exception as e:
        results["error"] = str(e)
        results["is_valid"] = None
        results["chi2_pass"] = None

    return results


def main():
    vectors_dir = sys.argv[1] if len(sys.argv) > 1 else "test_vectors"
    uni_dir = os.path.join(vectors_dir, "univariate")

    if not os.path.isdir(uni_dir):
        print(f"No vectors found at {uni_dir}. Run generate_test_vectors.py first.")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(uni_dir, "*.json")))
    print(f"Running SAGA baseline on {len(files)} univariate vectors...\n")

    all_results = []
    counts = {"good": {"pass": 0, "fail": 0, "error": 0},
              "bad": {"pass": 0, "fail": 0, "error": 0},
              "mediocre": {"pass": 0, "fail": 0, "error": 0}}

    for fpath in files:
        with open(fpath) as f:
            vector = json.load(f)

        result = run_univariate(vector)
        tier = vector["tier"]
        label = vector["label"]

        if result["error"]:
            status = "ERROR"
            counts[tier]["error"] += 1
        elif result.get("is_valid_extended", result["is_valid"]):
            status = "PASS"
            counts[tier]["pass"] += 1
        else:
            status = "FAIL"
            counts[tier]["fail"] += 1

        # For bad vectors, FAIL is correct (test detected the flaw)
        if tier == "bad":
            correct = "ok" if status == "FAIL" else "MISSED"
        elif tier == "good":
            correct = "ok" if status == "PASS" else "FALSE-ALARM"
        else:
            correct = ""

        flaw_type = vector["flaw"]["type"]
        ext_note = ""
        if not result["error"] and result.get("extended_pass") is not None:
            if not result["is_valid"] and result["is_valid"]:
                ext_note = ""
            elif result["is_valid"] and not result.get("is_valid_extended", True):
                ext_note = " [extended caught]"
        print(f"  [{status:>5}] {label:<45} flaw={flaw_type:<20} {correct}{ext_note}")

        all_results.append({
            "label": label,
            "tier": tier,
            "flaw": vector["flaw"],
            "status": status,
            "correct": correct,
            "results": result
        })

    # Summary
    print("\n--- Baseline Summary ---")
    print(f"{'Tier':>10} | {'Pass':>5} | {'Fail':>5} | {'Error':>5}")
    print("-" * 40)
    for tier in ["good", "bad", "mediocre"]:
        c = counts[tier]
        print(f"{tier:>10} | {c['pass']:>5} | {c['fail']:>5} | {c['error']:>5}")

    # Detection rate for bad vectors
    bad_total = counts["bad"]["pass"] + counts["bad"]["fail"] + counts["bad"]["error"]
    if bad_total > 0:
        detect_rate = counts["bad"]["fail"] / bad_total * 100
        print(f"\nBad vector detection rate: {counts['bad']['fail']}/{bad_total} = {detect_rate:.1f}%")

    # False alarm rate for good vectors
    good_total = counts["good"]["pass"] + counts["good"]["fail"] + counts["good"]["error"]
    if good_total > 0:
        false_alarm = counts["good"]["fail"] / good_total * 100
        print(f"Good vector false alarm rate: {counts['good']['fail']}/{good_total} = {false_alarm:.1f}%")

    # Save
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    outpath = "baseline_results.json"
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nDetailed results: {os.path.abspath(outpath)}")


if __name__ == "__main__":
    main()
