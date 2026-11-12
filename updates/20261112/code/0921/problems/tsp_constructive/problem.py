import numpy as np


class SearchStrategy:
    """Nearest-neighbor next-city policy; the construction loop is external."""

    def select_next_node(self, current_node, destination_node, unvisited_nodes, distance_matrix):
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
