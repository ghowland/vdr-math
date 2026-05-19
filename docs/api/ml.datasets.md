<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/datasets.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.datasets`
vdr.ml.datasets — Tokenization and dataset utilities. 

 from vdr.ml.datasets import tiny_text_dataset, one_hot 

 windows, vocab, inv_vocab = tiny_text_dataset("hello world", seq_len=3)  vec = one_hot(2, 10)  # Vec with 1 at index 2, 0 elsewhere 

All token ids are integers. One-hot vectors are exact VDR. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/datasets.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `build_vocab`

```python
build_vocab(tokens)
```

Build vocabulary mapping from unique tokens. 

I: list of string tokens O: dict {token: id} sorted by first appearance 

 vocab = build_vocab(["hello", "world", "hello"])  # {"hello": 0, "world": 1} 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/datasets.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `encode_tokens`

```python
encode_tokens(tokens, vocab)
```

Encode token strings to integer ids. 

I: list of strings, vocab dict O: list of ints 

 encode_tokens(["hello", "world"], vocab) -> [0, 1] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/datasets.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `decode_tokens`

```python
decode_tokens(ids, inv_vocab)
```

Decode integer ids to token strings. 

I: list of ints, inverse vocab dict O: list of strings 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/datasets.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `invert_vocab`

```python
invert_vocab(vocab)
```

Invert vocabulary: {token: id} -> {id: token}. 

I: vocab dict O: inverse dict 

 inv = invert_vocab({"hello": 0, "world": 1})  # {0: "hello", 1: "world"} 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/datasets.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sliding_windows`

```python
sliding_windows(ids, seq_len)
```

Create sliding window training examples from token id sequence. 

Each window is (context, target) where context has seq_len tokens and target is the next token. 

I: list of token ids, sequence length O: list of (context_list, target_int) tuples 

 sliding_windows([0, 1, 2, 3, 4], 3) 
    -> [([0,1,2], 3), ([1,2,3], 4)] 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/datasets.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `one_hot`

```python
one_hot(index, size)
```

One-hot vector: 1 at index, 0 elsewhere. 

I: index (int), vector size (int) O: Vec with VDR(1) at index and VDR(0) elsewhere 

 one_hot(2, 5) -> Vec([VDR(0), VDR(0), VDR(1), VDR(0), VDR(0)]) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/datasets.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `batchify_windows`

```python
batchify_windows(windows, vocab_size)
```

Convert sliding windows to training pairs with one-hot targets. 

I: list of (context, target) tuples, vocabulary size O: list of (context_ids, target_vec) tuples 

 batches = batchify_windows(windows, vocab_size=10) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/datasets.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tiny_text_dataset`

```python
tiny_text_dataset(text, seq_len)
```

Build a tiny text dataset from a string. 

Splits on whitespace, builds vocab, creates sliding windows. 

I: text string, sequence length O: (windows, vocab, inv_vocab) tuple 

 windows, vocab, inv_vocab = tiny_text_dataset(  "the cat sat on the mat", seq_len=2  ) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
