import unittest
import numpy as np

from tests.data.get_data import get_small_scaled_visium
from tests.testing_utils import initialize_testing
import spatialmuon as smu
import matplotlib.pyplot as plt

test_data_dir, DEBUGGING = initialize_testing()


class Graphs_TestClass(unittest.TestCase):
    def test_plot(self):
        s = get_small_scaled_visium()
        node_positions = s["visium"]["expression"].masks.untransformed_masks_centers[...]
        edge_indices = np.array([[a, b] for a, b in zip(*np.triu_indices(10)) if a != b])
        edge_features = np.random.rand(len(edge_indices))
        g = smu.Graph(
            untransformed_node_positions=node_positions,
            edge_indices=edge_indices,
            edge_features=edge_features,
            undirected=True
        )
        s['visium']['expression'].graph = g
        ##
        plt.style.use('dark_background')
        _, ax = plt.subplots(1, figsize=(10, 10))
        s['visium']['expression'].plot(0, ax=ax)
        s['visium']['expression'].graph.plot(node_colors=None, edge_colors='w', ax=ax)
        plt.show()
        plt.style.use('default')
        print("ooo")
        ##


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main(failfast=True)
    else:
        Graphs_TestClass().test_plot()
