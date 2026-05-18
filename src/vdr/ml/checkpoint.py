"""
vdr.ml.checkpoint — Exact model state save/load.

    from vdr.ml.checkpoint import save_model, load_parameters

    state = save_model(model)
    # JSON-serializable dict with all parameter values as VDR dicts

Every parameter saved as exact integer triple [V, D, R].
Reload produces bit-identical parameters on any platform.
"""

from __future__ import annotations
from typing import Dict, List, Any

from vdr.core import VDR
from vdr.linalg import Vec, Mat, vdr_to_dict, vdr_from_dict

__all__ = [
    "save_parameters",
    "load_parameters",
    "save_model",
    "load_model_parameters",
]


def save_parameters(params):
    """
    Save list of parameters to JSON-serializable dict.

    I: list of VecParam or MatParam
    O: dict with parameter names and values

    Each VDR value serialized as {"v": int, "d": int, "r": {...}}.
    Reload produces exact same VDR objects.

        state = save_parameters(model.parameters())
    """
    saved = []
    for i, p in enumerate(params):
        name = getattr(p, "name", None) or ("param_%d" % i)
        if hasattr(p.value, '_rows'):
            # MatParam
            rows = []
            for r in range(p.value.nrows):
                row = []
                for c in range(p.value.ncols):
                    row.append(vdr_to_dict(p.value[r, c]))
                rows.append(row)
            saved.append({
                "name": name,
                "type": "mat",
                "nrows": p.value.nrows,
                "ncols": p.value.ncols,
                "data": rows,
            })
        else:
            # VecParam
            data = [vdr_to_dict(p.value[i]) for i in range(len(p.value))]
            saved.append({
                "name": name,
                "type": "vec",
                "dim": len(p.value),
                "data": data,
            })
    return {"parameters": saved}


def load_parameters(saved):
    """
    Load parameter values from saved dict.

    I: dict from save_parameters
    O: list of (name, value) tuples where value is Vec or Mat

    Does NOT restore into model — caller assigns to model parameters.

        params = load_parameters(saved_state)
        for (name, value), param in zip(params, model.parameters()):
            param.value = value
    """
    result = []
    for entry in saved["parameters"]:
        name = entry["name"]
        if entry["type"] == "mat":
            rows = []
            for row_data in entry["data"]:
                row = [vdr_from_dict(d) for d in row_data]
                rows.append(row)
            result.append((name, Mat(rows)))
        else:
            data = [vdr_from_dict(d) for d in entry["data"]]
            result.append((name, Vec(data)))
    return result


def save_model(model):
    """
    Save entire model state.

    I: model with .parameters() method
    O: JSON-serializable dict

        state = save_model(model)
        import json
        json.dumps(state)  # works
    """
    return save_parameters(model.parameters())


def load_model_parameters(model, saved):
    """
    Load saved parameters into model.

    I: model with .parameters(), saved dict from save_model
    S: updates model parameter values in place

        load_model_parameters(model, saved_state)
    """
    loaded = load_parameters(saved)
    params = model.parameters()

    if len(loaded) != len(params):
        raise ValueError(
            "Parameter count mismatch: saved %d, model %d" % (
                len(loaded), len(params))
        )

    for (name, value), param in zip(loaded, params):
        param.value = value
