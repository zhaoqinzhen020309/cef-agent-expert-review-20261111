"""Task-neutral bounded result transport; domain feasibility belongs to the evaluator."""
import math

import numpy as np


def normalize_structured_solution(value, *, max_items=1_000_000, max_depth=32):
    """Copy primitive results, rejecting arbitrary objects and unbounded nesting.

    Suitable for subsets, assignments and other evaluator-defined outputs.
    Being serializable is not evidence that a solution is feasible or correct.
    Normalization is charged to the solver deadline by the caller.
    """
    remaining = max_items

    def visit(item, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > max_depth:
            raise ValueError("structured solution exceeds transport limits")
        if isinstance(item, np.ndarray):
            if item.size > remaining or item.ndim + depth > max_depth:
                raise ValueError("structured solution exceeds transport limits")
            if item.dtype.kind not in "biuf":
                raise ValueError("structured arrays must have primitive numeric values")
            return visit(item.tolist(), depth)
        if isinstance(item, np.generic):
            item = item.item()
        if item is None or type(item) in (bool, int):
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("structured solution contains a nonfinite number")
            return item
        if type(item) is str:
            remaining -= len(item)
            if remaining < 0:
                raise ValueError("structured solution exceeds transport limits")
            return item
        if type(item) in (tuple, list):
            if len(item) > remaining:
                raise ValueError("structured solution exceeds transport limits")
            return [visit(child, depth + 1) for child in item]
        if type(item) is dict:
            if len(item) > remaining:
                raise ValueError("structured solution exceeds transport limits")
            result = {}
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("structured solution keys must be strings")
                result[visit(key, depth + 1)] = visit(child, depth + 1)
            return result
        raise ValueError("structured solution contains an unsupported object")

    return visit(value, 0)
