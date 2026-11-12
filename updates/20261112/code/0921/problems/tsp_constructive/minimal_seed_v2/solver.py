"""Human engineering control based on the earlier Agent two-step rollout.

Explicitly NOT a new Agent discovery. Only next-city selection is implemented;
there is no completed-tour local search or access to evaluator references.
"""
import numpy as np


class SearchStrategy:
    FIRST_CHOICES = 8
    SECOND_CHOICES = 4

    def select_next_node(self, current_node, destination_node, unvisited_nodes, distance_matrix):
        nodes = np.sort(np.asarray(unvisited_nodes, dtype=np.int64))
        count = len(nodes)
        if count == 0:
            return int(destination_node)
        if count == 1:
            return int(nodes[0])
        distances = np.asarray(distance_matrix)
        local = distances[np.ix_(nodes, nodes)]
        first = np.argsort(distances[current_node, nodes], kind="stable")[:self.FIRST_CHOICES]
        initial = local[first].copy()
        initial[np.arange(len(first)), first] = np.inf
        second = np.argsort(initial, axis=1, kind="stable")[:, :min(self.SECOND_CHOICES, count - 1)]
        first = np.repeat(first, second.shape[1])
        current = second.ravel()
        rows = np.arange(len(first))
        visited = np.zeros((len(first), count), dtype=bool)
        visited[rows, first] = True
        visited[rows, current] = True
        costs = distances[current_node, nodes[first]] + local[first, current]
        # K bounded trajectories advance together. O(K*m^2) per decision,
        # O(K*n^3) for the whole construction, O(n^2 + K*n) auxiliary space.
        for _ in range(count - 2):
            options = local[current].copy()
            options[visited] = np.inf
            next_nodes = np.argmin(options, axis=1)
            costs += local[current, next_nodes]
            visited[rows, next_nodes] = True
            current = next_nodes
        costs += distances[nodes[current], destination_node]
        return int(nodes[first[int(np.argmin(costs))]])
