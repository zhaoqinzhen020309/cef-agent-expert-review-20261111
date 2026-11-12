"""Starting generator adapted from ASRO Appendix H.1, arXiv:2601.22896.

Geometric families and default ranges are prior-method input, recorded as such.
"""
import numpy as np


def generate(n_cities, seed, parameters):
    rng = np.random.default_rng(seed)
    family = parameters.get('family', 'mixture')
    if family == 'mixture':
        family = rng.choice(['uniform','clustered','grid','ring','two_scale'], p=[.30,.25,.15,.15,.15])
    if family == 'uniform': points = rng.random((n_cities, 2))
    elif family == 'clustered':
        k = int(rng.integers(3, 9)); centers = rng.random((k, 2))
        points = centers[rng.integers(k, size=n_cities)] + rng.normal(0, rng.uniform(.02,.08), (n_cities,2))
    elif family == 'grid':
        side = int(np.ceil(np.sqrt(n_cities)))
        grid = np.array([(x,y) for x in np.linspace(0,1,side) for y in np.linspace(0,1,side)])
        points = grid[rng.choice(len(grid),n_cities,replace=False)] + rng.uniform(-.05,.05,(n_cities,2))
    elif family == 'ring':
        angles = rng.uniform(0,2*np.pi,n_cities)
        radius = rng.uniform(rng.uniform(.2,.4),rng.uniform(.5,.7),n_cities)
        points = .5 + np.column_stack((np.cos(angles),np.sin(angles))) * radius[:,None]
    elif family == 'two_scale':
        count = int(.7*n_cities); small = rng.random((n_cities-count,2))*rng.uniform(.05,.15)
        corner = rng.integers(0,2,2); small = np.where(corner,1-small,small)
        points = np.concatenate((rng.random((count,2)),small)); rng.shuffle(points)
    else: raise ValueError('unknown geometric family')
    return np.clip(points,0,1).tolist()
