import unittest
import numpy as np

import spatialmuon
from tests.data.get_data import get_small_scaled_visium, get_small_imc
from tests.testing_utils import initialize_testing
import spatialmuon as smu
import matplotlib.pyplot as plt
import matplotlib.cm
import tempfile
import shutil
import os

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

    def test_subgraph_of_neighbors(self):
        s = get_small_imc()
        g = s["imc"]["masks"].compute_knn_graph(k=20, max_distance_in_units=20)
        max_distance = 20
        sub_g0, center_index0, original_indices0 = g.subgraph_of_neighbors(
            node_indices=10, subset_method="proximity", max_distance=max_distance
        )
        sub_g1, center_index1, original_indices1 = g.subgraph_of_neighbors(
            node_indices=3, k=None, subset_method="knn"
        )
        sub_g2_3, center_indices2_3, original_indices2_3 = g.subgraph_of_neighbors(
            node_indices=[3, 4], k=None, subset_method="knn"
        )
        assert len(sub_g2_3[1].edge_features) != len(sub_g1.edge_features) or not np.alltrue(
            sub_g2_3[1].edge_features == sub_g1.edge_features
        )
        assert np.alltrue(sub_g2_3[0].edge_features == sub_g1.edge_features)

        ##
        _, ax = plt.subplots()
        g.plot(ax=ax, node_colors="w", edge_colors="w", edge_size=0.5)

        sub_g0.plot(ax=ax, node_colors="r", edge_colors="r")
        ax.scatter(*sub_g0.untransformed_node_positions[center_index0].tolist(), c="g", s=100)
        circle = plt.Circle(
            sub_g0.untransformed_node_positions[center_index0].tolist(),
            max_distance,
            edgecolor="r",
            facecolor=(0.0, 0.0, 0.0, 0.0),
        )
        ax.add_patch(circle)

        sub_g1.plot(ax=ax, node_colors="r", edge_colors="r")
        ax.scatter(*sub_g1.untransformed_node_positions[center_index1].tolist(), c="g", s=100)

        original_locations = g.transformed_node_positions
        ax.scatter(
            *original_locations[original_indices0[center_index0], :].tolist(), s=500, marker="o"
        )

        ax.set_aspect("equal")
        plt.show()
        ##

    def test_load_from_file(self):
        fpath_imc = test_data_dir / "small_imc.h5smu"
        with tempfile.TemporaryDirectory() as td:
            des = os.path.join(td, "small_imc.h5smu")
            shutil.copy(fpath_imc, des)
            s = spatialmuon.SpatialMuData(backing=des)
            g = s["imc"]["masks"].compute_knn_graph(k=20, max_distance_in_units=20)
            s["imc"]["masks"].graph = g
            s1 = smu.SpatialMuData(backing=s.backing.filename)


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main(failfast=True)
    else:
        # Graphs_TestClass().test_plot()
        # Graphs_TestClass().test_knn_graph()
        # Graphs_TestClass().test_rbfk_graph()
        # Graphs_TestClass().test_proximity_graph()
        # Graphs_TestClass().test_graph_plot_with_cmap()
        # Graphs_TestClass().test_to_networkx_on_large_graphs()
        Graphs_TestClass().test_subgraph_of_neighbors()
        # Graphs_TestClass().test_load_from_file()
