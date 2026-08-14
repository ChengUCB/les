#!/usr/bin/env python
"""
Run the whole LES test suite with one command.

    python test/run_all.py              # everything
    python test/run_all.py -k dipole    # only tests whose name matches 'dipole'
    python test/run_all.py -v           # also show each test's output

The suite is a mix of styles, so this runner handles them uniformly, and says
which kind each target is:

  PASS  the target checks its own numbers -- pytest assertions, or a script that
        compares against a reference and exits nonzero when it disagrees
  RAN   the script ran to completion without raising, but asserts nothing. That
        is a smoke test: it catches import errors, shape errors and exceptions,
        not wrong numbers. Never read it as "verified".

Exit code is 0 only if every selected target reached PASS or RAN, so it works in
CI too; a RAN target can only fail by raising.
Set PYTHONPATH to the repo's src/ (this runner does it automatically) to make
sure the checked-out LES is tested rather than an installed copy.
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")

# pytest targets: assert-based, reported as one row each
PYTEST_TARGETS = [
    "test/test_ewald_realspace",
    "src/les/tests",
]

# script targets: (path, extra argv, checks_itself)
SCRIPT_TARGETS = [
    ("test/test_ewald_vectorized_physics.py", [], True),           # vectorized physics vs legacy/main
    ("test/test_ewald_vectorized_compile.py", [], True),           # torch.compile + AOTInductor gates
    ("test/test_ewald_vectorization_compilation.py", [], True),    # legacy vs vectorized
    ("test/test_torch_all_features.py", [], True),                 # TorchScript, all terms
    ("test/test_torch_dipole.py", [], False),
    ("test/test_les_class.py", [], False),
    ("test/test_bec.py", [], False),
    ("test/test_grad.py", [], False),
    ("test/test_quick.py", [], False),
    ("test/test_ewald_triclinic.py", [], False),
    ("test/test_ewald_real.py", ["42"], False),                   # needs a seed
    ("test/test_ewald_real_dipoles.py", ["42"], False),           # needs a seed
    ("test/test_ewald_realspace/print_q_u_Q_induced.py", [], False),
]


def env():
    e = os.environ.copy()
    # src/ so the checked-out LES is tested, test/ for the shared _vec_harness
    extra = SRC + os.pathsep + os.path.join(REPO, "test")
    e["PYTHONPATH"] = extra + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    return e


def run(cmd, cwd):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env(), capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--filter", default="", help="substring match on the test name")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each test's output")
    args = ap.parse_args()

    rows, failed = [], []

    def selected(name):
        return args.filter in name

    for target in PYTEST_TARGETS:
        if not selected(target):
            continue
        rc, out, dt = run([sys.executable, "-m", "pytest", target, "-q"], REPO)
        # pull pytest's own summary line, e.g. "38 passed"
        tail = [l for l in out.strip().splitlines() if "passed" in l or "failed" in l or "error" in l]
        rows.append((target, "PASS" if rc == 0 else "FAIL",
                     tail[-1].strip() if tail else "", dt))
        if rc != 0:
            failed.append((target, out))
        if args.verbose:
            print(out)

    for path, extra, checks in SCRIPT_TARGETS:
        if not selected(path):
            continue
        if not os.path.exists(os.path.join(REPO, path)):
            rows.append((path, "SKIP", "missing", 0.0))
            continue
        rc, out, dt = run([sys.executable, os.path.basename(path)] + extra,
                          os.path.join(REPO, os.path.dirname(path)))
        if rc != 0:
            status, note = "FAIL", ""
            failed.append((path, out))
        elif checks:
            status, note = "PASS", "comparisons asserted"
        else:
            status, note = "RAN", "no assertions -- smoke only"
        rows.append((path, status, note, dt))
        if args.verbose:
            print(out)

    width = max(len(r[0]) for r in rows) if rows else 20
    print("\n" + "=" * (width + 34))
    for name, status, note, dt in rows:
        print(f"{status:<4}  {name:<{width}}  {dt:5.1f}s  {note}")
    print("=" * (width + 34))
    n_ran = sum(1 for r in rows if r[1] == "RAN")
    if n_ran:
        print(f"note: {n_ran} target(s) reported RAN -- they exercise the code but "
              f"check no numbers, so they cannot detect a wrong result.")

    if failed:
        for name, out in failed:
            print(f"\n----- output of failing {name} (tail) -----")
            print("\n".join(out.strip().splitlines()[-25:]))
        print(f"\n{len(failed)}/{len(rows)} test targets FAILED")
        sys.exit(1)
    n_pass = sum(1 for r in rows if r[1] == "PASS")
    print(f"{n_pass}/{len(rows)} targets verified their numbers, "
          f"{n_ran} ran without checking, none failed")


if __name__ == "__main__":
    main()
