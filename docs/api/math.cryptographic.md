<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/cryptographic.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.cryptographic`
vdr.math.cryptographic — Exact modular arithmetic and cryptographic primitives. 

 from vdr.math.cryptographic import rsa_keygen, rsa_encrypt, rsa_decrypt 

 n, e, d = rsa_keygen(61, 53, 17)  c = rsa_encrypt(42, e, n)  m = rsa_decrypt(c, d, n)  # 42 exactly 

All operations exact integer modular arithmetic. Float categorically excluded from this domain. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/cryptographic.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mod_exp`

```python
mod_exp(base, exp, mod)
```

Fast modular exponentiation: base^exp mod mod. 

I: integers base, exp (non-negative), mod (positive) O: integer result 

Uses repeated squaring. O(log exp) multiplications. 

 mod_exp(2, 10, 1000) -> 24 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/cryptographic.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `extended_gcd`

```python
extended_gcd(a, b)
```

Extended Euclidean algorithm. 

I: integers a, b O: (gcd, x, y) where a*x + b*y = gcd 

 extended_gcd(35, 15) -> (5, 1, -2) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/cryptographic.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mod_inverse`

```python
mod_inverse(a, m)
```

Modular multiplicative inverse: a^(-1) mod m. 

I: integer a, modulus m O: integer x such that (a * x) % m == 1 

 mod_inverse(17, 3120) -> 2753 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/cryptographic.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `chinese_remainder`

```python
chinese_remainder(remainders, moduli)
```

Chinese Remainder Theorem reconstruction. 

I: list of remainders [r1, r2, ...], list of pairwise coprime moduli [m1, m2, ...] O: integer x such that x = ri (mod mi) for all i, 0 <= x < product(mi) 

 chinese_remainder([2, 3, 2], [3, 5, 7]) -> 23 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/cryptographic.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `euler_totient`

```python
euler_totient(n)
```

Euler's totient function phi(n). 

I: positive integer n O: integer count of values 1..n coprime to n 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/cryptographic.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rsa_keygen`

```python
rsa_keygen(p, q, e)
```

RSA key generation. 

I: primes p, q, public exponent e O: (n, e, d) where n = p*q, d = e^(-1) mod phi(n) 

 rsa_keygen(61, 53, 17) -> (3233, 17, 2753) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/cryptographic.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rsa_encrypt`

```python
rsa_encrypt(m, e, n)
```

RSA encryption: c = m^e mod n. 

I: plaintext integer m, public exponent e, modulus n O: ciphertext integer c 

 rsa_encrypt(42, 17, 3233) -> some ciphertext 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/cryptographic.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rsa_decrypt`

```python
rsa_decrypt(c, d, n)
```

RSA decryption: m = c^d mod n. 

I: ciphertext integer c, private exponent d, modulus n O: plaintext integer m 

 rsa_decrypt(ciphertext, 2753, 3233) -> 42 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/cryptographic.py#L187"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `baby_giant`

```python
baby_giant(g, h, p)
```

Baby-step giant-step discrete logarithm. 

Find x such that g^x = h (mod p). 

I: generator g, target h, prime modulus p O: integer x, or raises ValueError if not found 

 baby_giant(2, 8, 13) -> 3  (because 2^3 = 8 mod 13) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/cryptographic.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_prime`

```python
is_prime(n)
```

Simple primality test. 

I: integer n O: bool 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/cryptographic.py#L245"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `miller_rabin_deterministic`

```python
miller_rabin_deterministic(n)
```

Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981. 

Uses the first 13 primes as witnesses, which is sufficient for all numbers below the bound above. 

I: integer n >= 2 O: bool (True if prime) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
