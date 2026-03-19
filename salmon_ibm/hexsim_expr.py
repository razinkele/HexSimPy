"""HexSim expression DSL translator.

Translates HexSim expression strings to Python-evaluable form.
Only quote references need regex; function names are injected into the eval namespace.

MUST NOT import from salmon_ibm.accumulators (circular import risk).
Only import: re, numpy, ast.
"""
from __future__ import annotations

import re
import numpy as np


_translate_cache: dict[str, str] = {}


def translate_hexsim_expr(expr: str) -> str:
    """Translate HexSim expression DSL to Python-evaluable string.

    Transformations:
    1. 'single quoted' → _g["single quoted"]  (global variable lookup)
    2. "double quoted" → _a["double quoted"]   (accumulator lookup)
    3. Cond → _cond                            (sign-of-difference semantics: test > 0 = true)

    Both quote types are processed in a single regex pass to prevent
    double-substitution (e.g., single-quote output containing double-quotes
    being re-matched by the double-quote pattern).
    """
    cached = _translate_cache.get(expr)
    if cached is not None:
        return cached

    def _replace_quotes(m: re.Match) -> str:
        if m.group(1) is not None:
            # Single-quoted global: 'name'
            return f'_g["{m.group(1)}"]'
        else:
            # Double-quoted accumulator: "name"
            return f'_a["{m.group(2)}"]'

    # Match either 'single-quoted' or "double-quoted" in one pass
    result = re.sub(r"'([^']+)'|\"([^\"]+)\"", _replace_quotes, expr)
    # Rename Cond to _cond (sign-of-difference semantics: test > 0 = true)
    result = re.sub(r'\bCond\b', '_cond', result)
    _translate_cache[expr] = result
    return result


def build_hexsim_namespace(globals_dict, acc_dict, rng, n_masked):
    """Build eval namespace with HexSim DSL functions + data.

    Parameters
    ----------
    globals_dict : dict mapping global variable names to float values
    acc_dict : dict mapping accumulator names to masked float64 arrays
    rng : numpy random Generator
    n_masked : int, number of masked agents (= mask.sum())
    """
    ns = {
        # Data lookups
        '_g': globals_dict,
        '_a': acc_dict,
        # HexSim functions (injected by name — Python parser handles nesting)
        '_cond': lambda test, t, f: np.where(np.asarray(test) > 0, t, f),
        'Floor': np.floor,
        'Pow': np.power,
        'Exp': np.exp,
        'Max': np.maximum,
        'Min': np.minimum,
        'GasDev': lambda: rng.standard_normal(n_masked),
        'Rand': lambda: rng.random(n_masked),
        # Standard math (backward compat with existing _SAFE_MATH)
        'np': np,
        'sqrt': np.sqrt, 'abs': np.abs, 'exp': np.exp,
        'log': np.log, 'log10': np.log10,
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'minimum': np.minimum, 'maximum': np.maximum,
        'clip': np.clip, 'where': np.where,
        'pi': np.pi, 'e': np.e,
    }
    return ns
