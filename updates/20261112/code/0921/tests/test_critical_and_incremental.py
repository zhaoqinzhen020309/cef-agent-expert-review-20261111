from pathlib import Path
import importlib.util
import sys
import time
import numpy as np

from agent_ahd.scripts.prepare_minimal_cvrp_incremental import build
from agent_ahd.scripts.prepare_minimal_fssp_critical import variants
from agent_ahd.scripts.profile_fssp_neighborhoods import instrument
from agent_ahd.scripts.prepare_minimal_cvrp_star import cached_star
from agent_ahd.scripts.prepare_cvrp_capacity_repairs import repaired
from agent_ahd.scripts.prepare_minimal_fssp_blocks import variants as block_variants
from agent_ahd.scripts.prepare_minimal_fssp_reconstruction import variants as reconstruction_variants
from agent_ahd.scripts.prepare_cvrp_final_optimization import optimize
from agent_ahd.evaluators.manifest_evaluator import objective

ROOT = Path(__file__).resolve().parents[1]


def load(tmp_path, name, text):
    path = tmp_path / (name + '.py')
    path.write_text(text, encoding='utf-8')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_deletion_bounds_match_independent_makespan(tmp_path):
    parent = (ROOT / 'problems/fssp/minimal_repair_seed_v1/solver.py').read_text()
    module = load(tmp_path, 'critical_bound_test', variants(parent)['critical_filter'])
    rng = np.random.default_rng(4490)
    for n, m in ((1, 1), (9, 4), (21, 10)):
        p = rng.random((n, m)) * 10
        seq = rng.permutation(n)
        f, b, costs = np.zeros((n + 1, m + 1)), np.zeros((n + 1, m + 1)), np.empty(n + 1)
        relief = module.removal_relief(seq, n, p, f, b)
        value = module.makespan(seq, n, p)
        for j in range(n):
            remaining = np.delete(seq, j)
            expected = value - module.makespan(remaining, n - 1, p)
            np.testing.assert_allclose(relief[seq[j]], expected, atol=1e-10)
        got = module.local_search(seq, n, value, p, f, b, costs, True)
        assert sorted(seq) == list(range(n))
        np.testing.assert_allclose(got, objective('fssp', p, seq.tolist()), atol=1e-10)
        assert got <= value + 1e-10


def test_host_instrumentation_preserves_seeded_fixed_work(tmp_path):
    source = (ROOT / 'problems/fssp/minimal_repair_seed_v1/solver.py').read_text()
    original = load(tmp_path, 'original_probe_test', source)
    probe = load(tmp_path, 'instrumented_probe_test', instrument(source))
    p = np.random.default_rng(4491).integers(1, 40, (12, 4)).astype(float)
    outcomes = []
    metrics = np.zeros((2, 6))
    for mod, extra in ((original, ()), (probe, (metrics,))):
        f, b, costs = np.zeros((13, 5)), np.zeros((13, 5)), np.empty(13)
        seq, value = mod.initialize(p, 729, f, b, costs, True, *extra)
        records = [(seq.copy(), value)]
        for _ in range(8):
            seq, value = mod.step(seq, p, 4, f, b, costs, True, True, *extra)
            records.append((seq.copy(), value))
        outcomes.append(records)
    for (a, av), (b, bv) in zip(*outcomes):
        np.testing.assert_array_equal(a, b)
        assert av == bv
    assert metrics[:, 2].sum() >= metrics[:, 3:5].sum() > 0


def test_incremental_route_state_matches_full_recomputation_after_moves(tmp_path):
    source = (ROOT / 'runs/minimal_campaign_v3/jobs/cvrp/selected_artifact/selected/solver.py').read_text()
    mod = load(tmp_path, 'incremental_cvrp_test', build(source))
    refresh = mod.SearchStrategy._refresh_routes
    rng = np.random.default_rng(4492)
    demands = rng.integers(1, 9, 61).astype(float)
    demands[0] = 0
    routes = [list(range(i, i + 5)) for i in range(1, 61, 5)]
    def blank():
        return [np.zeros(61, dtype=np.int64) for _ in range(4)] + [np.zeros(61), np.zeros(len(routes))]
    state = blank()
    state[-1] = refresh(routes, demands, *state)
    for turn in range(100):
        a, b = rng.choice(len(routes), 2, replace=False)
        if turn % 3 == 0 and routes[a]:
            routes[b].insert(0, routes[a].pop())
        elif turn % 3 == 1 and routes[a] and routes[b]:
            routes[a][0], routes[b][-1] = routes[b][-1], routes[a][0]
        else:
            ia, ib = len(routes[a]) // 2, len(routes[b]) // 2
            routes[a], routes[b] = routes[a][:ia] + routes[b][ib:], routes[b][:ib] + routes[a][ia:]
        state[-1] = refresh(routes, demands, *state, affected=(a, b))
        expected = blank()
        expected[-1] = refresh(routes, demands, *expected)
        for actual, full in zip(state, expected):
            np.testing.assert_array_equal(actual, full)


def test_cached_swap_star_matches_exhaustive_pair_insertion(tmp_path):
    source = (ROOT / 'runs/minimal_campaign_v3/jobs/cvrp/selected_artifact/selected/solver.py').read_text()
    mod = load(tmp_path, 'cached_star_test', cached_star(build(source)))
    solver = mod.SearchStrategy()
    rng = np.random.default_rng(4493)
    for _ in range(20):
        n = 21
        xy = rng.integers(0, 30, (n, 2))
        dist = np.round(np.linalg.norm(xy[:, None] - xy[None, :], axis=2))
        demands = rng.integers(1, 8, n).astype(float)
        demands[0] = 0
        permutation = rng.permutation(n - 1) + 1
        routes = [permutation[i:i + 4].tolist() for i in range(0, n - 1, 4)]
        capacity = max(sum(demands[c] for c in r) for r in routes) + 2
        rid, pos, prev, nxt = [np.zeros(n, dtype=int) for _ in range(4)]
        prefix = np.zeros(n)
        loads = solver._refresh_routes(routes, demands, rid, pos, prev, nxt, prefix, np.zeros(len(routes)))
        for a in (1, 5, 13):
            targets = np.array([c for c in permutation if rid[c] != rid[a]])[:12]
            actual = solver._swap_star(a, targets, routes, rid, prev, nxt, loads, dist, demands, capacity, time.time() + 10)
            expected = None
            ra = int(rid[a])
            for b in targets:
                r = int(rid[b])
                if loads[ra] - demands[a] + demands[b] > capacity or loads[r] - demands[b] + demands[a] > capacity:
                    continue
                aa, bb = [v for v in routes[ra] if v != a], [v for v in routes[r] if v != b]
                old = solver._route_cost(routes[ra], dist) + solver._route_cost(routes[r], dist)
                costs_a = [solver._route_cost(aa[:i] + [int(b)] + aa[i:], dist) for i in range(len(aa) + 1)]
                costs_b = [solver._route_cost(bb[:i] + [a] + bb[i:], dist) for i in range(len(bb) + 1)]
                ia, ib = int(np.argmin(costs_a)), int(np.argmin(costs_b))
                delta = costs_a[ia] + costs_b[ib] - old
                if delta < -1e-9 and (expected is None or delta < expected[0]):
                    expected = (delta, r, aa[:ia] + [int(b)] + aa[ia:], bb[:ib] + [a] + bb[ib:])
            assert actual == expected


def test_tail_exchange_keeps_cut_customer_in_capacity_constraint(tmp_path):
    path = ROOT / 'runs/minimal_campaign_v3/jobs/cvrp/nodes/node_0002/versions/b97e4c678cfb52356e4d82d690f6c6afa59c536c79b41116a8a06535467e7bd9/solver.py'
    source = path.read_text()
    old = load(tmp_path, 'bad_tail_test', source).SearchStrategy()
    fixed = load(tmp_path, 'fixed_tail_test', repaired(source, 'node_0002')).SearchStrategy()
    d = np.full((5, 5), 100.)
    np.fill_diagonal(d, 0)
    d[0, 1:] = d[1:, 0] = 1
    for a, b, cost in ((1, 2, 10), (3, 4, 10), (1, 4, 1), (3, 2, 1)):
        d[a, b] = d[b, a] = cost
    demands = np.array([0., 8, 2, 2, 8])
    outputs = []
    for solver in (old, fixed):
        solver.LOCAL_PASSES = 1
        routes, loads = solver._local_search([[1, 2], [3, 4]], np.array([10., 10.]),
                                            d, demands, 10., time.time() + 2)
        outputs.append(max(sum(demands[c] for c in r) for r in routes))
    assert outputs[0] > 10
    assert outputs[1] <= 10


def test_selected_segment_carries_its_load_and_refreshes_route_ids(tmp_path):
    path = ROOT / 'runs/minimal_campaign_v3/jobs/cvrp/nodes/node_0001/versions/036380f65bc0a6f00e2353119b77f0eb3327c36ef66d8dc9ca673dc0e9f39c68/solver.py'
    solver = load(tmp_path, 'fixed_segment_test', repaired(path.read_text(), 'node_0001')).SearchStrategy()
    rng = np.random.default_rng(4494)
    for _ in range(12):
        n = 25
        xy = rng.random((n, 2))
        dist = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
        demands = rng.integers(1, 10, n).astype(float)
        demands[0] = 0
        routes = [list(range(i, i + 4)) for i in range(1, n, 4)]
        caps = np.array([sum(demands[c] for c in r) for r in routes])
        capacity = max(caps) + 3
        rid = np.full(n, -1, dtype=int)
        for r, route in enumerate(routes):
            rid[route] = r
        neighbors = solver._build_neighbors(dist, n)
        solver._relocate_or_opt_best(routes, caps, dist, demands, capacity, neighbors, rid, time.time() + 1)
        np.testing.assert_allclose(caps, [sum(demands[c] for c in route) for route in routes])
        assert max(caps) <= capacity
        for r, route in enumerate(routes):
            assert all(rid[c] == r for c in route)


def test_guided_block_moves_keep_permutation_and_exact_objective(tmp_path):
    source = (ROOT / 'problems/fssp/minimal_repair_seed_v1/solver.py').read_text()
    module = load(tmp_path, 'critical_block_test', block_variants(source)['critical_boundary_blocks'])
    rng = np.random.default_rng(4495)
    for n, m in ((2, 1), (9, 4), (20, 7)):
        p = rng.integers(1, 40, (n, m)).astype(float)
        seq = rng.permutation(n)
        f, b, costs = np.zeros((n + 1, m + 1)), np.zeros((n + 1, m + 1)), np.empty(n + 1)
        initial = objective('fssp', p, seq.tolist())
        value = module.block_search(seq, initial, p, f, b, costs, 3)
        assert value == objective('fssp', p, seq.tolist()) <= initial


def test_reconstruction_controls_return_exact_costs_on_real_processing_times(tmp_path):
    source = (ROOT / 'problems/fssp/minimal_repair_seed_v1/solver.py').read_text()
    rng = np.random.default_rng(4496)
    for name, text in reconstruction_variants(source).items():
        if name == 'parent':
            continue
        module = load(tmp_path, name + '_test', text)
        for n, m in ((1, 1), (12, 4)):
            p = rng.random((n, m)) * 20
            f, b, costs = np.zeros((n + 1, m + 1)), np.zeros((n + 1, m + 1)), np.empty(n + 1)
            seq, value = module.initialize(p, 14, f, b, costs, True)
            for _ in range(4):
                seq, value = module.step(seq, p, 4, f, b, costs, True, True)
                assert sorted(seq) == list(range(n))
                np.testing.assert_allclose(value, objective('fssp', p, seq.tolist()), atol=1e-10)


def test_final_cvrp_optimization_preserves_fixed_work_output(tmp_path):
    source = (ROOT / 'runs/minimal_cvrp_confirmation_v1/selected_artifact/solver.py').read_text()
    modules = (load(tmp_path, 'confirmed_cvrp_test', source), load(tmp_path, 'optimized_cvrp_test', optimize(source)))
    class Clock:
        def __init__(self): self.tick = 0
        def time(self):
            self.tick += 1
            return self.tick
    rng = np.random.default_rng(4497)
    for n in (12, 30, 60):
        xy = rng.random((n, 2))
        demands = rng.integers(1, 10, n).astype(float)
        demands[0] = 0
        item = {'distance_matrix': np.linalg.norm(xy[:, None] - xy[None, :], axis=2),
                'demands': demands, 'capacity': 25.}
        for seed in (17, 29):
            outputs = []
            for module in modules:
                module.time = Clock()
                routes = module.SearchStrategy().solve(item, 3000, 1000, seed)
                objective('cvrp', item, routes)
                outputs.append(routes)
            assert outputs[0] == outputs[1]
