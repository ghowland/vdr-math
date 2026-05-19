<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml`
vdr.ml — Exact machine learning primitives. 

 from vdr.ml.softmax import softmax  from vdr.ml.nn import Linear, ReLU, Sequential  from vdr.ml.attention import self_attention  from vdr.ml.optim import SGD 

Every operation exact VDR rational. Softmax sums to exactly 1. Gradients exact via chain rule. No float drift in training loops. 





---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
