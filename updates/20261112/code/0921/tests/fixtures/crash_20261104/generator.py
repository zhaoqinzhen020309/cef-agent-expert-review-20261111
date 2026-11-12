import numpy as np


def generate(n_cities, seed, parameters):
    rng = np.random.default_rng(seed)
    family = parameters.get('family', 'mixture')
    if family == 'mixture':
        family = rng.choice(
            ['uniform', 'clustered', 'grid', 'ring', 'two_scale', 'clustered_high',
             'spiral', 'thin_ring', 'grid_aniso'],
            p=[0.06, 0.20, 0.14, 0.10, 0.12, 0.14, 0.08, 0.08, 0.08]
        )
    if family == 'uniform':
        points = rng.random((n_cities, 2))
    elif family == 'clustered':
        k = int(rng.integers(3, 9))
        centers = rng.random((k, 2))
        sigma = rng.uniform(0.02, 0.08)
        points = centers[rng.integers(k, size=n_cities)] + rng.normal(0, sigma, (n_cities, 2))
    elif family == 'clustered_high':
        k = int(rng.integers(10, 20))
        centers = rng.random((k, 2))
        sigma = rng.uniform(0.005, 0.03)
        points = centers[rng.integers(k, size=n_cities)] + rng.normal(0, sigma, (n_cities, 2))
    elif family == 'grid':
        side = int(np.ceil(np.sqrt(n_cities)))
        grid = np.array([(x, y) for x in np.linspace(0, 1, side) for y in np.linspace(0, 1, side)])
        points = grid[rng.choice(len(grid), n_cities, replace=False)] + rng.uniform(-0.03, 0.03, (n_cities, 2))
    elif family == 'ring':
        angles = rng.uniform(0, 2 * np.pi, n_cities)
        radius = rng.uniform(rng.uniform(0.2, 0.4), rng.uniform(0.5, 0.7), n_cities)
        points = 0.5 + np.column_stack((np.cos(angles), np.sin(angles))) * radius[:, None]
    elif family == 'two_scale':
        count = int(0.7 * n_cities)
        small = rng.random((n_cities - count, 2)) * rng.uniform(0.02, 0.08)
        corner = rng.integers(0, 2, 2)
        small = np.where(corner, 1 - small, small)
        points = np.concatenate((rng.random((count, 2)), small))
        rng.shuffle(points)
    elif family == 'spiral':
        t = np.linspace(0, rng.uniform(2.5, 5) * np.pi, n_cities)
        r = np.linspace(0.01, rng.uniform(0.4, 0.48), n_cities)
        jitter = rng.normal(0, 0.012, (n_cities, 2))
        points = np.column_stack((0.5 + r * np.cos(t), 0.5 + r * np.sin(t))) + jitter
    elif family == 'thin_ring':
        angles = rng.uniform(0, 2 * np.pi, n_cities)
        base_r = rng.uniform(0.35, 0.45)
        radius = base_r + rng.normal(0, 0.008, n_cities)
        points = 0.5 + np.column_stack((np.cos(angles), np.sin(angles))) * radius[:, None]
    elif family == 'grid_aniso':
        side = int(np.ceil(np.sqrt(n_cities)))
        sx = rng.uniform(0.5, 1.0)
        sy = rng.uniform(0.5, 1.0)
        grid = np.array([(x * sx, y * sy) for x in np.linspace(0, 1, side) for y in np.linspace(0, 1, side)])
        pts = grid[rng.choice(len(grid), n_cities, replace=False)]
        pts = pts + rng.uniform(-0.04, 0.04, (n_cities, 2))
        my = pts[:, 1].max() if pts[:, 1].max() > 0 else 1.0
        mx = pts[:, 0].max() if pts[:, 0].max() > 0 else 1.0
        pts[:, 0] /= mx
        pts[:, 1] /= my
        points = pts
    else:
        raise ValueError('unknown geometric family')
    points = np.clip(points, 0, 1)
    seen = {}
    out = []
    for p in points:
        key = (round(float(p[0]), 12), round(float(p[1]), 12))
        if key in seen:
            cnt = seen[key]
            seen[key] += 1
            while True:
                angle = (cnt * 0.618033988749895) % (2 * np.pi)
                r = 1e-10 * (1 + cnt)
                q = np.array([p[0] + r * np.cos(angle), p[1] + r * np.sin(angle)])
                q = np.clip(q, 0, 1)
                key2 = (round(float(q[0]), 12), round(float(q[1]), 12))
                if key2 not in seen:
                    break
                cnt += 1
            seen[key2] = 1
            p = q
        else:
            seen[key] = 1
        out.append(p)
    return [[float(x), float(y)] for x, y in out]
