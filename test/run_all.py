#!/usr/bin/env python
"""
Run the whole LES test suite with one command.

    python test/run_all.py              # everything
    python test/run_all.py -k dipole    # only tests whose name matches 'dipole'
    python test/run_all.py -v           # also show each test's output

The suite is a mix of styles, so this runner handles them uniformly:
  * pytest directories (assert-based unit tests)
  * plain scripts that print results (treated as pass unless they raise)
  * scripts that need a CLI seed argument

Exit code is 0 only if every selected test passed, so it works in CI too.
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

# script targets: (path, extra argv). A script fails only if it raises / exits nonzero.
SCRIPT_TARGETS = [
    ("test/test_ewald_vectorized_physics.py", []),                # vectorized physics vs legacy/main
    ("test/test_ewald_vectorized_compile.py", []),                # torch.compile + AOTInductor gates
    ("test/test_ewald_vectorization_compilation.py", []),         # legacy vs vectorized
    ("test/test_torch.py", []),
    ("test/test_torch_dipole.py", []),
    ("test/test_torch_all_features.py", []),                       # TorchScript, all terms
    ("test/test_les_class.py", []),
    ("test/test_bec.py", []),
    ("test/test_grad.py", []),
    ("test/test_quick.py", []),
    ("test/test_ewald_triclinic.py", []),
    ("test/test_ewald_real.py", ["42"]),                          # needs a seed
    ("test/test_ewald_real_dipoles.py", ["42"]),                  # needs a seed
    ("test/test_ewald_realspace/print_q_u_Q_induced.py", []),
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
        rows.append((target, rc == 0, tail[-1].strip() if tail else "", dt))
        if rc != 0:
            failed.append((target, out))
        if args.verbose:
            print(out)

    for path, extra in SCRIPT_TARGETS:
        if not selected(path):
            continue
        if not os.path.exists(os.path.join(REPO, path)):
            rows.append((path, True, "skipped (missing)", 0.0))
            continue
        rc, out, dt = run([sys.executable, os.path.basename(path)] + extra,
                          os.path.join(REPO, os.path.dirname(path)))
        note = ""
        if rc == 0:
            # surface the summary line of the gate-style harness when present
            for line in out.splitlines():
                if "checks passed" in line or "SUMMARY" in line:
                    note = "gates passed"
        rows.append((path, rc == 0, note, dt))
        if rc != 0:
            failed.append((path, out))
        if args.verbose:
            print(out)

    width = max(len(r[0]) for r in rows) if rows else 20
    print("\n" + "=" * (width + 26))
    for name, ok, note, dt in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {dt:5.1f}s  {note}")
    print("=" * (width + 26))

    if failed:
        for name, out in failed:
            print(f"\n----- output of failing {name} (tail) -----")
            print("\n".join(out.strip().splitlines()[-25:]))
        print(f"\n{len(failed)}/{len(rows)} test targets FAILED")
        sys.exit(1)
    print(f"all {len(rows)} test targets passed")


if __name__ == "__main__":
    main()
