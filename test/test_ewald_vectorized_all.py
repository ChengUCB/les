"""
Everything in one run: the physics gates and the compile/AOTInductor gates.

Convenience entry point for a manual full check. It just calls the two focused
tests, so there is no duplicated gate logic:

    test_ewald_vectorized_physics.py   eager correctness, seconds
    test_ewald_vectorized_compile.py   torch.compile + AOTInductor, minutes

Each phase uses its own system size (the physics gates want enough atoms to be
meaningful, the codegen gates want small kernels), so the size is set explicitly
here rather than inherited from whichever module was imported last.

run_all.py runs the two focused tests separately and does NOT run this file, to
avoid doing the expensive compile work twice.

Run:
    PYTHONPATH=src:test python test/test_ewald_vectorized_all.py
"""
import sys

import _vec_harness as H
import test_ewald_vectorized_compile as compile_gates
import test_ewald_vectorized_physics as physics


def _phase(name, fn, n_per):
    print(f"\n##################### {name} #####################")
    H.N_PER = n_per
    try:
        fn()
        return True
    except SystemExit as exc:                 # main() exits non-zero on failure
        return (exc.code or 0) == 0


def main():
    ok_physics = _phase("PHYSICS", physics.main, [5, 6, 7])
    ok_compile = _phase("COMPILE", compile_gates.main, [2, 3])

    print("\n##################### OVERALL #####################")
    print(f"  physics: {'PASS' if ok_physics else 'FAIL'}")
    print(f"  compile: {'PASS' if ok_compile else 'FAIL'}")
    if not (ok_physics and ok_compile):
        sys.exit(1)
    print("everything passed")


if __name__ == "__main__":
    main()
