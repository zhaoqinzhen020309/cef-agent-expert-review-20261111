import pytest
from agent_ahd.core.implementation_preflight import validate_implementation_source


def test_detects_undefined_name_inside_jit_kernel_without_execution():
    with pytest.raises(ValueError, match="undefined global name 'machines'"):
        validate_implementation_source("from numba import njit\n@njit\ndef local_search(p):\n    return machines + len(p)\n")


def test_detects_missing_helper_argument_before_compilation():
    with pytest.raises(ValueError, match="helper step call"):
        validate_implementation_source("def step(p, count, neutral):\n    return p\ndef solve(p):\n    return step(p, 2)\n")


def test_accepts_defaults_keywords_closures_and_deferred_global_definitions():
    validate_implementation_source("""
def step(p, count=2, *, neutral=False):
    return p[:count]
def solve(p):
    def inner():
        return step(p, neutral=True)
    return inner() + [GLOBAL_VALUE]
GLOBAL_VALUE = 3
""")


def test_does_not_guess_dynamic_decorators_splats_or_shadowed_signatures():
    validate_implementation_source("""
def wrapper(f):
    return lambda: 3
@wrapper
def dynamic(x):
    return x
def helper(a, b):
    return a
def solve(helper, args):
    return helper(1), dynamic(), helper(*args)
""")


def test_dead_conditional_expression_does_not_create_a_false_name_error():
    validate_implementation_source("def solve(j):\n    return j in cand_set_dummy if False else False\n")


def test_detects_never_initialized_numeric_buffer():
    with pytest.raises(ValueError, match="uninitialized NumPy buffer 'primary'"):
        validate_implementation_source('import numpy as np\ndef f(n):\n    primary = np.empty(n)\n    return np.argsort(primary[:n])\n')


def test_allows_explicit_writes_and_opaque_initializing_helpers():
    validate_implementation_source('''
import numpy as np
def fill(x):
    x[:] = 1
def f(n):
    a = np.empty(n)
    fill(a)
    b = np.empty(n)
    b.fill(2)
    c = np.empty(n)
    for i in range(n):
        c[i] = i
    d = np.empty(n)
    alias = d
    alias[:] = 4
    return a[0] + b[0] + c[0] + d[0]
''')


def test_jit_argsort_kind_check_does_not_reject_python_stable_sort():
    validate_implementation_source('import numpy as np\ndef f(x):\n    return np.argsort(x, kind="stable")\n')
    with pytest.raises(ValueError, match='Numba argsort does not support'):
        validate_implementation_source('import numpy as np\nfrom numba import njit\n@njit\ndef f(x):\n    return np.argsort(x, kind="stable")\n')
    validate_implementation_source('import numpy as np\nfrom numba import njit\n@njit\ndef f(x):\n    return np.argsort(x, kind="mergesort")\n')
