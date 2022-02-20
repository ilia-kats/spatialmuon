import unittest
import numpy as np

from tests.data.get_data import get_small_scaled_visium, get_small_imc
from tests.testing_utils import initialize_testing
import spatialmuon as smu
import matplotlib.pyplot as plt
import matplotlib.cm

plt.style.use("dark_background")

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
            undirected=True,
        )
        s["visium"]["expression"].graph = g

        _, ax = plt.subplots(1, figsize=(10, 10))
        s["visium"]["expression"].plot(0, ax=ax)
        s["visium"]["expression"].graph.plot(node_colors=None, edge_colors="w", ax=ax)
        plt.show()

    def test_knn_graph(self):
        s = get_small_scaled_visium()
        g = s["visium"]["expression"].compute_knn_graph(k=10)
        s["visium"]["expression"].graph = g

        _, ax = plt.subplots(1, figsize=(10, 10))
        bb = s["visium"]["expression"].bounding_box
        s["visium"]["image"].plot(ax=ax, bounding_box=bb)
        s["visium"]["expression"].graph.plot(node_colors=None, edge_colors="k", ax=ax)
        s["visium"]["expression"].plot(0, ax=ax)
        plt.show()
        pass

    # Radial Basis Funciton (RBF) kernel graph
    def test_rbfk_graph(self):
        s = get_small_scaled_visium()
        g = s["visium"]["expression"].compute_rbfk_graph(
            length_scale_in_units=55, max_distance_in_units=200
        )
        s["visium"]["expression"].graph = g

        _, ax = plt.subplots(1, figsize=(10, 10))
        s["visium"]["expression"].graph.plot(node_colors=None, edge_colors="w", ax=ax)
        s["visium"]["expression"].plot(0, ax=ax)
        plt.show()

    def test_proximity_graph(self):
        s = get_small_imc()
        g = s["imc"]["masks"].compute_proximity_graph(max_distance_in_units=20)
        s["imc"]["masks"].graph = g

        _, ax = plt.subplots(1, figsize=(10, 10))
        s["imc"]["masks"].plot(0, ax=ax)
        s["imc"]["masks"].graph.plot(node_colors=None, edge_colors="w", ax=ax)
        plt.show()

    def test_graph_plot_with_cmap(self):
        s = get_small_imc()
        g0 = s["imc"]["masks"].compute_knn_graph(k=20, max_distance_in_units=20)
        g1 = s["imc"]["masks"].compute_proximity_graph(max_distance_in_units=20)

        _, axes = plt.subplots(1, 2)
        g0.plot(
            node_colors=None,
            edge_cmap=matplotlib.cm.get_cmap("viridis"),
            ax=axes[0],
        )
        g1.plot(
            node_colors=None,
            edge_cmap=matplotlib.cm.get_cmap("viridis"),
            ax=axes[1],
        )
        plt.show()

    # this test used to fail because the function Graph.to_networkx() was changing the order of edges
    def test_to_networkx_on_large_graphs(self):
        g = smu.Graph.disconnected_graph(untransformed_node_positions=np.random.rand(10000, 2))
        g.compute_knn_edges(k=10)
        g.to_networkx()


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main(failfast=True)
    else:
        Graphs_TestClass().test_plot()
        Graphs_TestClass().test_knn_graph()
        Graphs_TestClass().test_rbfk_graph()
        Graphs_TestClass().test_proximity_graph()
        Graphs_TestClass().test_graph_plot_with_cmap()
        Graphs_TestClass().test_to_networkx_on_large_graphs()
