"""Register bounded-neighborhood engineering controls using training only."""
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]

EXTENSION = '''

class SearchStrategy(SavingsBase):
    NEIGHBORS = 24
    TAIL_EXCHANGE = True

    def solve(self, instance, deadline, iteration_limit, seed):
        start = time.time()
        remaining = deadline - start
        construction_deadline = min(deadline, start + max(.25, remaining * .3))
        routes = super().solve(instance, construction_deadline, iteration_limit, seed)
        dist = np.asarray(instance['distance_matrix'], dtype=float)
        demands = np.asarray(instance['demands'], dtype=float)
        capacity = float(instance['capacity'])
        n = len(dist)
        stop = deadline - .12
        if n < 3 or time.time() >= stop:
            return routes
        k = min(self.NEIGHBORS + 1, n - 1)
        neighbors = np.argpartition(dist[:, 1:], k - 1, axis=1)[:, :k] + 1
        rid = np.full(n, -1, dtype=np.int64)
        position = np.zeros(n, dtype=np.int64)
        previous = np.zeros(n, dtype=np.int64)
        following = np.zeros(n, dtype=np.int64)
        prefix = np.zeros(n)
        loads = np.zeros(len(routes))
        rng = np.random.default_rng(seed)

        def refresh():
            loads.fill(0)
            for r, route in enumerate(routes):
                total = 0.0
                for i, customer in enumerate(route):
                    total += demands[customer]
                    rid[customer], position[customer] = r, i
                    previous[customer] = route[i - 1] if i else 0
                    following[customer] = route[i + 1] if i + 1 < len(route) else 0
                    prefix[customer] = total
                loads[r] = total

        refresh()
        for sweep in range(min(iteration_limit, 40)):
            improved = False
            for route in routes:
                if time.time() >= stop:
                    return [r for r in routes if r]
                if len(route) > 2:
                    old_cost = self._route_cost(route, dist)
                    if self._two_opt_route(route, dist, stop) < old_cost - 1e-9:
                        improved = True
            refresh()
            for customer in rng.permutation(n - 1) + 1:
                if time.time() >= stop:
                    return [r for r in routes if r]
                a = int(customer)
                ra = int(rid[a])
                targets = neighbors[a]
                targets = targets[(targets != a) & (rid[targets] != ra)]
                if not len(targets):
                    continue
                rb = rid[targets]
                pa, na = previous[a], following[a]
                pb, nb = previous[targets], following[targets]
                remove = dist[pa, na] - dist[pa, a] - dist[a, na]
                deltas = np.full((4, len(targets)), np.inf)
                can_relocate = loads[rb] + demands[a] <= capacity + 1e-9
                deltas[0] = np.where(can_relocate, remove + dist[pb, a] + dist[a, targets] - dist[pb, targets], np.inf)
                deltas[1] = np.where(can_relocate, remove + dist[targets, a] + dist[a, nb] - dist[targets, nb], np.inf)
                can_swap = ((loads[ra] - demands[a] + demands[targets] <= capacity + 1e-9)
                            & (loads[rb] - demands[targets] + demands[a] <= capacity + 1e-9))
                deltas[2] = np.where(can_swap,
                    dist[pa, targets] + dist[targets, na] - dist[pa, a] - dist[a, na]
                    + dist[pb, a] + dist[a, nb] - dist[pb, targets] - dist[targets, nb], np.inf)
                if self.TAIL_EXCHANGE:
                    can_exchange = ((prefix[a] + loads[rb] - prefix[targets] <= capacity + 1e-9)
                                    & (prefix[targets] + loads[ra] - prefix[a] <= capacity + 1e-9))
                    deltas[3] = np.where(can_exchange, dist[a, nb] + dist[targets, na] - dist[a, na] - dist[targets, nb], np.inf)
                index = int(np.argmin(deltas))
                move, column = divmod(index, len(targets))
                if deltas[move, column] >= -1e-9:
                    continue
                b = int(targets[column])
                r = int(rid[b])
                ia, ib = int(position[a]), int(position[b])
                if move < 2:
                    del routes[ra][ia]
                    routes[r].insert(ib + move, a)
                elif move == 2:
                    routes[ra][ia], routes[r][ib] = b, a
                else:
                    tail_a, tail_b = routes[ra][ia + 1:], routes[r][ib + 1:]
                    routes[ra] = routes[ra][:ia + 1] + tail_b
                    routes[r] = routes[r][:ib + 1] + tail_a
                refresh()
                improved = True
            if not improved:
                break
        return [route for route in routes if route]
'''


def main():
    root = ROOT / "runs/minimal_cvrp_granular_v2"
    parent = ROOT / "problems/cvrp/minimal_seed_v2/solver.py"
    original = parent.read_text()
    base = original.replace("    def solve(", "    def _savings_construct(", 1)
    if base == original:
        raise ValueError("missing declared savings parent")
    protocol = ROOT / "data/bundles/minimal_v1/cvrp/protocol.json"
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    registry = {"problem": "cvrp", "protocol": str(protocol), "protocol_sha256": sha(protocol),
                "evidence_type": "human_engineering_training_controls; not a new Agent result", "parent_sha256": sha(parent),
                "budget": {"n_instances": 5, "time_limit": 2, "ite_max": 1000, "seeds": [71400, 71401]}, "sources": {}}
    variants = {"baseline": original}
    for neighbors, exchange in ((12, True), (24, True), (24, False)):
        variants[f"granular_k{neighbors}_tails{int(exchange)}"] = base + EXTENSION.replace("class SearchStrategy(SavingsBase):\n", "").replace("super().solve(", "self._savings_construct(").replace("NEIGHBORS = 24", f"NEIGHBORS = {neighbors}").replace("TAIL_EXCHANGE = True", f"TAIL_EXCHANGE = {exchange}")
    for name, source in variants.items():
        path = root / "sources" / name / "solver.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(source, encoding="utf-8")
        registry["sources"][name] = {"path": str(path), "sha256": sha(path)}
    (root / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(root / "registry.json")


if __name__ == "__main__":
    main()
