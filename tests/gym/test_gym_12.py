"""Gym 12 — Cryptographic primitives. 37/37 in VDR-2."""

import pytest
from vdr.math.cryptographic import (
    mod_exp, extended_gcd, mod_inverse, chinese_remainder,
    rsa_keygen, rsa_encrypt, rsa_decrypt, baby_giant,
)


class TestModExp:
    def test_basic(self):
        assert mod_exp(2, 10, 1000) == 24

    def test_fermat(self):
        """Fermat's little theorem: a^(p-1) = 1 mod p."""
        for p in [7, 11, 13, 17, 23]:
            for a in range(2, p):
                assert mod_exp(a, p - 1, p) == 1


class TestExtGCD:
    def test_basic(self):
        g, x, y = extended_gcd(35, 15)
        assert g == 5
        assert 35 * x + 15 * y == 5

    def test_coprime(self):
        g, x, y = extended_gcd(17, 13)
        assert g == 1


class TestModInverse:
    def test_basic(self):
        inv = mod_inverse(17, 3120)
        assert (17 * inv) % 3120 == 1

    def test_non_invertible_raises(self):
        with pytest.raises(ValueError):
            mod_inverse(6, 12)


class TestCRT:
    def test_basic(self):
        x = chinese_remainder([2, 3, 2], [3, 5, 7])
        assert x % 3 == 2
        assert x % 5 == 3
        assert x % 7 == 2


class TestRSA:
    def test_roundtrip(self):
        n, e, d = rsa_keygen(61, 53, 17)
        assert n == 3233
        for m in [42, 0, 100, 3232]:
            c = rsa_encrypt(m, e, n)
            recovered = rsa_decrypt(c, d, n)
            assert recovered == m

    def test_five_messages(self):
        n, e, d = rsa_keygen(61, 53, 17)
        messages = [65, 111, 200, 1000, 3000]
        for m in messages:
            assert rsa_decrypt(rsa_encrypt(m, e, n), d, n) == m


class TestBabyGiant:
    def test_basic(self):
        # 2^3 = 8 mod 13
        assert baby_giant(2, 8, 13) == 3

    def test_another(self):
        # 3^x = 13 mod 17 -> 3^4 = 81 = 4*17+13 -> x=4
        assert baby_giant(3, 13, 17) == 4
