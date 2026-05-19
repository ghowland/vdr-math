<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.transformer`
vdr.ml.transformer — Exact transformer architecture. 

 from vdr.ml.transformer import TransformerLM 

 model = TransformerLM(embedding, [block], output_proj)  logits = model.forward_logits([0, 1, 2]) 

Every attention weight sums to exactly 1. Every gradient exact. No float drift across layers or sequence positions. All weight matrices projected to basis frame via to_qbasis. 



---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Embedding`
Lookup table embedding: integer token id -> Vec. 

 emb = Embedding.from_ints([[1,0,0],[0,1,0],[0,0,1]])  v = emb.lookup(1)  # Vec([VDR(0), VDR(1), VDR(0)]) 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(table, name=None)
```

I: table as list of Vec 


---

#### <kbd>property</kbd> dim





---

#### <kbd>property</kbd> vocab_size







---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_fracs`

```python
from_fracs(rows, name=None)
```

Construct from fraction tuples. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_ints`

```python
from_ints(rows, name=None)
```

Construct from integer lists. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `lookup`

```python
lookup(idx)
```

Look up embedding vector for token id. 

I: integer token id O: Vec 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `lookup_many`

```python
lookup_many(ids)
```

Look up embedding vectors for sequence of token ids. 

I: list of integer token ids O: list of Vec 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_qbasis`

```python
to_qbasis(bits=None)
```

Project all embedding vectors onto Q basis. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FFNBlock`
Feed-forward network block: linear -> relu -> linear. 

Common in transformer architectures. 

 ffn = FFNBlock(  Linear.from_ints([[1,2],[3,4],[5,6]], [0,0,0]),  Linear.from_ints([[1,0],[0,1],[1,1]], [0,0]),  ) 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(l1, l2)
```








---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward(grad_out)
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

x -> linear -> relu -> linear. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `parameters`

```python
parameters()
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_qbasis`

```python
to_qbasis(bits=None)
```

Project both linear layers to basis frame. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zero_grad`

```python
zero_grad()
```






---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TransformerBlock`
Single transformer block: self-attention + FFN. 

For each position:  h = self_attention(Q, K, V)  (with optional causal mask)  out = ffn(h) 

Simplified: no layer norm, no residual connection in v1. These can be added as the identity + output pattern. 

 block = TransformerBlock(Wq, Wk, Wv, Wo, ffn)  outputs = block.forward(token_vecs) 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(Wq, Wk, Wv, Wo, ffn, causal=True, exp_depth=16)
```

I: projection matrices Wq, Wk, Wv, Wo as Mat,  ffn as FFNBlock, causal flag, exp depth 




---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(xs)
```

Forward pass through transformer block. 

I: xs as list of Vec (sequence of hidden states) O: list of Vec (transformed hidden states) 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L216"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_with_cache`

```python
forward_with_cache(xs)
```

Forward pass returning attention intermediate values for inspection. 

O: (outputs, Q, K, V, attn_outputs) 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L184"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `parameters`

```python
parameters()
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_qbasis`

```python
to_qbasis(bits=None)
```

Project all weight matrices and FFN to basis frame. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zero_grad`

```python
zero_grad()
```






---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L250"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TransformerLM`
Complete transformer language model. 

embedding -> transformer blocks -> output projection -> logits 

 model = TransformerLM(embedding, [block1, block2], output_proj)  logits = model.forward_logits([0, 1, 2, 3]) 

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L260"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(embedding, blocks, output_proj)
```

I: Embedding, list of TransformerBlock, Linear output projection 




---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L280"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `embed`

```python
embed(token_ids)
```

Embed token ids into vectors. 

I: list of integer token ids O: list of Vec 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L289"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_hidden`

```python
forward_hidden(token_ids)
```

Forward pass through embedding and transformer blocks. 

I: list of integer token ids O: list of Vec (final hidden states) 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L301"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_logits`

```python
forward_logits(token_ids)
```

Full forward pass producing logits. 

I: list of integer token ids O: list of Vec (logit vectors, one per position) 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L311"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_logits_with_cache`

```python
forward_logits_with_cache(token_ids)
```

Forward pass with cached intermediates for inspection. 

O: (logits, hidden_states) 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `parameters`

```python
parameters()
```





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L321"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_qbasis`

```python
to_qbasis(bits=None)
```

Project entire model to basis frame: embedding, all blocks (weights + FFN), and output projection. 

---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/transformer.py#L275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zero_grad`

```python
zero_grad()
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
