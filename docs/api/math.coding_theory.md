<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `math.coding_theory`
vdr.math.coding_theory — Exact finite field and error correction. 

 from vdr.math.coding_theory import hamming74_encode, hamming74_correct, gf_inv 

 codeword = hamming74_encode([1, 0, 1, 1])  corrected = hamming74_correct([1, 0, 1, 1, 0, 1, 1])  # single-bit fix 

All field arithmetic exact modular integer. Float categorically excluded. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gf_add`

```python
gf_add(a, b, p)
```

Addition in GF(p): (a + b) mod p. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gf_sub`

```python
gf_sub(a, b, p)
```

Subtraction in GF(p): (a - b) mod p. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gf_mul`

```python
gf_mul(a, b, p)
```

Multiplication in GF(p): (a * b) mod p. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gf_inv`

```python
gf_inv(a, p)
```

Multiplicative inverse in GF(p): a^(-1) mod p. 

I: element a (nonzero), prime p O: integer x such that (a * x) % p == 1 

 gf_inv(3, 7) -> 5  (because 3*5 = 15 = 1 mod 7) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gf_div`

```python
gf_div(a, b, p)
```

Division in GF(p): a * b^(-1) mod p. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gf_pow`

```python
gf_pow(a, n, p)
```

Exponentiation in GF(p): a^n mod p. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `gf_poly_eval`

```python
gf_poly_eval(coeffs, x, p)
```

Evaluate polynomial over GF(p) using Horner's method. 

I: coefficient list [a0, a1, ...] (constant first), point x, prime p O: p(x) mod p 

 gf_poly_eval([1, 2, 3], 4, 11) -> (1 + 2*4 + 3*16) mod 11 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `hamming74_encode`

```python
hamming74_encode(data)
```

Encode 4-bit data word to 7-bit Hamming(7,4) codeword. 

I: list of 4 bits [d0, d1, d2, d3] O: list of 7 bits [d0, d1, d2, d3, p0, p1, p2] 

 hamming74_encode([1, 0, 1, 1]) -> [1, 0, 1, 1, 0, 0, 0] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `hamming74_syndrome`

```python
hamming74_syndrome(received)
```

Compute syndrome of received 7-bit word. 

I: list of 7 bits O: integer syndrome (0-7), 0 means no error 

 hamming74_syndrome([1, 0, 1, 1, 0, 0, 0]) -> 0 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `hamming74_correct`

```python
hamming74_correct(received)
```

Correct single-bit error in received 7-bit Hamming(7,4) codeword. 

I: list of 7 bits (possibly with one error) O: corrected list of 7 bits 

Syndrome maps to error position:  0 -> no error  1-7 -> flip bit at position (syndrome - 1) 

 hamming74_correct([1, 1, 1, 1, 0, 0, 0])  # if original was [1, 0, 1, 1, 0, 0, 0], corrects bit 1 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `hamming_distance`

```python
hamming_distance(a, b)
```

Hamming distance between two bit vectors. 

I: two lists of bits (same length) O: integer count of differing positions 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `hamming_weight`

```python
hamming_weight(a)
```

Hamming weight of a bit vector. 

I: list of bits O: integer count of 1s 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `min_distance`

```python
min_distance(codewords)
```

Minimum Hamming distance of a code. 

I: list of codewords (each a list of bits) O: minimum distance between any two distinct codewords 

 codewords = [hamming74_encode(d) for d in all_4bit_words]  min_distance(codewords) -> 3 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L226"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `repetition_decode`

```python
repetition_decode(bits, n)
```

Majority-vote decoding for repetition code. 

I: received bits (list of int), repetition factor n O: decoded bit (0 or 1) 

 repetition_decode([1, 0, 1], 3) -> 1 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/math/coding_theory.py#L241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_checksum`

```python
compute_checksum(data, p)
```

Simple polynomial checksum over GF(p). 

I: list of integer data values, prime p O: checksum integer in GF(p) 

Evaluates data as polynomial at x=2 in GF(p). 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
