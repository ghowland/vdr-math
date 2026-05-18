"""
Test: verify core operators are now basis-frame-safe.
Run: python tests/test_basis_safe_ops.py
"""

from vdr import VDR
from vdr.basis import set_default, get_default, to_qbasis

def check(label, actual_d, expected_d):
    ok = "OK" if actual_d == expected_d else "FAIL"
    print("  %-40s D = %-20d [%s]" % (label, actual_d, ok))
    return actual_d == expected_d

def main():
    set_default(32)
    denom = 2 ** get_default()
    passed = 0
    failed = 0
    total = 0

    a = to_qbasis(VDR(1, 3))
    b = to_qbasis(VDR(1, 7))
    c = to_qbasis(VDR(2, 5))

    print("basis: 2^%d = %d\n" % (get_default(), denom))

    # --- core * chain ---
    print("=== core * chain ===")
    x = a
    for i in range(10):
        x = x * b
        total += 1
        if check("mul step %d" % (i + 1), x.d, denom):
            passed += 1
        else:
            failed += 1
    print()

    # --- core / chain ---
    print("=== core / chain ===")
    x = a
    for i in range(10):
        x = x / b
        total += 1
        if check("div step %d" % (i + 1), x.d, denom):
            passed += 1
        else:
            failed += 1
    print()

    # --- core + chain ---
    print("=== core + chain ===")
    x = a
    for i in range(10):
        x = x + c
        total += 1
        if check("add step %d" % (i + 1), x.d, denom):
            passed += 1
        else:
            failed += 1
    print()

    # --- core - chain ---
    print("=== core - chain ===")
    x = a
    for i in range(10):
        x = x - c
        total += 1
        if check("sub step %d" % (i + 1), x.d, denom):
            passed += 1
        else:
            failed += 1
    print()

    # --- mixed: basis value + plain VDR ---
    print("=== mixed ops (basis + non-basis) ===")
    x = to_qbasis(VDR(1, 3))
    y = VDR(1, 3)

    z = x + y
    total += 1
    if check("basis + VDR(1,3)", z.d, denom):
        passed += 1
    else:
        failed += 1

    z = x - y
    total += 1
    if check("basis - VDR(1,3)", z.d, denom):
        passed += 1
    else:
        failed += 1

    z = x * y
    total += 1
    # non-basis operand, D will differ — this is expected
    print("  %-40s D = %-20d [INFO: mixed, no basis guarantee]" % (
        "basis * VDR(1,3)", z.d))

    z = x / y
    total += 1
    print("  %-40s D = %-20d [INFO: mixed, no basis guarantee]" % (
        "basis / VDR(1,3)", z.d))
    print()

    # --- non-basis ops should be unaffected ---
    print("=== non-basis ops (should be unchanged) ===")
    p = VDR(1, 3)
    q = VDR(1, 7)

    z = p + q
    total += 1
    expected = 21
    if check("VDR(1,3) + VDR(1,7) -> D=21", z.d, expected):
        passed += 1
    else:
        failed += 1

    z = p * q
    total += 1
    expected = 21
    if check("VDR(1,3) * VDR(1,7) -> D=21", z.d, expected):
        passed += 1
    else:
        failed += 1

    z = p / q
    total += 1
    expected = 3
    if check("VDR(1,3) / VDR(1,7) -> D=3", z.d, expected):
        passed += 1
    else:
        failed += 1
    print()

    # --- value correctness spot checks ---
    print("=== value correctness ===")

    x = to_qbasis(VDR(1, 3))
    y = to_qbasis(VDR(1, 7))
    z = x * y
    expected_float = 1.0 / 21.0
    actual_float = z.to_float()
    err = abs(actual_float - expected_float)
    total += 1
    ok = err < 1e-8
    if ok:
        passed += 1
    else:
        failed += 1
    print("  %-40s err = %.2e  [%s]" % (
        "basis(1/3) * basis(1/7) ≈ 1/21", err, "OK" if ok else "FAIL"))

    x = to_qbasis(VDR(1, 3))
    z = x * x * x * x * x
    expected_float = (1.0 / 3.0) ** 5
    actual_float = z.to_float()
    err = abs(actual_float - expected_float)
    total += 1
    ok = err < 1e-8
    if ok:
        passed += 1
    else:
        failed += 1
    print("  %-40s err = %.2e  [%s]" % (
        "basis(1/3)^5 ≈ 1/243", err, "OK" if ok else "FAIL"))

    x = to_qbasis(VDR(22, 7))
    y = to_qbasis(VDR(7, 22))
    z = x * y
    expected_float = 1.0
    actual_float = z.to_float()
    err = abs(actual_float - expected_float)
    total += 1
    ok = err < 1e-8
    if ok:
        passed += 1
    else:
        failed += 1
    print("  %-40s err = %.2e  [%s]" % (
        "basis(22/7) * basis(7/22) ≈ 1", err, "OK" if ok else "FAIL"))
    print()

    # --- change basis mid-run ---
    print("=== basis change mid-run ===")
    set_default(16)
    denom16 = 2 ** get_default()
    a16 = to_qbasis(VDR(1, 3))
    b16 = to_qbasis(VDR(1, 7))

    z = a16 * b16
    total += 1
    if check("after set_default(16): mul", z.d, denom16):
        passed += 1
    else:
        failed += 1

    z = a16 + b16
    total += 1
    if check("after set_default(16): add", z.d, denom16):
        passed += 1
    else:
        failed += 1

    z = a16 / b16
    total += 1
    if check("after set_default(16): div", z.d, denom16):
        passed += 1
    else:
        failed += 1

    # mix old 2^32 value with new 2^16 value
    z = a + a16
    total += 1
    print("  %-40s D = %-20d [INFO: cross-basis mix]" % (
        "2^32 val + 2^16 val", z.d))
    print()

    # --- summary ---
    print("=" * 60)
    print("passed: %d / %d" % (passed, total))
    if failed:
        print("FAILED: %d" % failed)
    else:
        print("all checks passed")


if __name__ == "__main__":
    main()
    