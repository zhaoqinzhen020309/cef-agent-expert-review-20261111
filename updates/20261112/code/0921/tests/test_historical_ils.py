import time

import numpy as np
import pytest

from scripts.benchmark_historical_ils import load, SOURCE, TARGET


@pytest.mark.parametrize("n", [3, 8, 30, 60])
@pytest.mark.parametrize("rounded", [False, True])
def test_historical_control_preserves_neighborhood_order(n, rounded):
    old, new = load(SOURCE), load(TARGET)
    rng = np.random.default_rng(91 + n)
    xy = rng.random((n, 2))
    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    if rounded:
        d = np.round(d * 10)
    neighbours = np.argsort(d, axis=1)[:, 1:min(n, 33)]
    route = rng.permutation(n).tolist()
    for method in ("_two_opt", "_relocate", "_or_opt_two", "_local_search"):
        expected = getattr(old, method)(route[:], d, neighbours, time.time() + 120)
        actual = getattr(new, method)(route[:], d, neighbours, time.time() + 120)
        assert expected == actual
    assert old.solve(d, time.time() + 120, 5, 11) == new.solve(d, time.time() + 120, 5, 11)
