<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/checkpoint.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `ml.checkpoint`
vdr.ml.checkpoint — Exact model state save/load. 

 from vdr.ml.checkpoint import save_model, load_parameters 

 state = save_model(model)  # JSON-serializable dict with all parameter values as VDR dicts 

Every parameter saved as exact integer triple [V, D, R]. Reload produces bit-identical parameters on any platform. 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/checkpoint.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_parameters`

```python
save_parameters(params)
```

Save list of parameters to JSON-serializable dict. 

I: list of VecParam or MatParam O: dict with parameter names and values 

Each VDR value serialized as {"v": int, "d": int, "r": {...}}. Reload produces exact same VDR objects. 

 state = save_parameters(model.parameters()) 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/checkpoint.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_parameters`

```python
load_parameters(saved)
```

Load parameter values from saved dict. 

I: dict from save_parameters O: list of (name, value) tuples where value is Vec or Mat 

Does NOT restore into model — caller assigns to model parameters. 

 params = load_parameters(saved_state)  for (name, value), param in zip(params, model.parameters()):  param.value = value 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/checkpoint.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_model`

```python
save_model(model)
```

Save entire model state. 

I: model with .parameters() method O: JSON-serializable dict 

 state = save_model(model)  import json  json.dumps(state)  # works 


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/ml/checkpoint.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_model_parameters`

```python
load_model_parameters(model, saved)
```

Load saved parameters into model. 

I: model with .parameters(), saved dict from save_model S: updates model parameter values in place 

 load_model_parameters(model, saved_state) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
