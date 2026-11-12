import os
import pickle as pkl

import numpy as np


class GetData:
    """Loads pre-generated TSP instances with known optimal tour costs.

    The training set (TSPAEL64.pkl) holds 64 random 100-node Euclidean TSP
    instances together with their optimal tour costs, used to compute the
    optimality gap during evolution.
    """

    def __init__(self, n_instance, datafile=None, instance_ids=None):
        self.n_instance = n_instance
        if datafile is None:
            datafile = os.path.join(os.path.dirname(__file__),
                                    'TrainingData', 'TSPAEL64.pkl')
        self.datafile = datafile
        self.instance_ids = tuple(instance_ids or ())

    def load_instances(self):
        with open(self.datafile, 'rb') as f:
            data = pkl.load(f)
        indices = self._indices(
            len(data['distance_matrix']),
            data.get('instance_ids'),
        )
        coords = [np.asarray(data['coordinate'][index]) for index in indices]
        instances = [np.asarray(data['distance_matrix'][index]) for index in indices]
        opt_costs = [data['cost'][index] for index in indices]
        return coords, instances, opt_costs

    def resolved_instance_ids(self):
        if self.instance_ids:
            return list(self.instance_ids)
        return [
            f"train_{index:04d}"
            for index in self._indices(self.n_instance, None)
        ]

    def _indices(self, available, stored_instance_ids):
        if self.instance_ids:
            if stored_instance_ids is not None:
                positions = {
                    str(instance_id): index
                    for index, instance_id in enumerate(stored_instance_ids)
                }
                unknown = set(self.instance_ids) - set(positions)
                if unknown:
                    raise ValueError(
                        f"unknown training instance ids: {sorted(unknown)}"
                    )
                return [positions[instance_id] for instance_id in self.instance_ids]
            indices = []
            for instance_id in self.instance_ids:
                try:
                    index = int(str(instance_id).rsplit("_", 1)[1])
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"invalid training instance id: {instance_id}") from exc
                if index < 0 or index >= available:
                    raise IndexError(f"training instance id out of range: {instance_id}")
                indices.append(index)
            return indices
        return list(range(min(self.n_instance, available)))
