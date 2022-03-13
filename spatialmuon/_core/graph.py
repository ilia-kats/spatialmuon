from __future__ import annotations

import math
from enum import Enum, auto
from typing import Optional, Union, Literal, TYPE_CHECKING, Tuple, Dict, List

import h5py
import matplotlib.axes
import matplotlib.cm
import matplotlib.collections
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from anndata._io.utils import read_attribute, write_attribute
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

from spatialmuon._core.backing import BackableObject
from spatialmuon._core.bounding_box import BoundingBoxable, BoundingBox
from spatialmuon.utils import (
    _get_hdf5_attribute,
    ColorsType,
    ColorType,
    handle_categorical_plot,
    get_color_array_rgba,
    apply_alpha,
)

if TYPE_CHECKING:
    pass


class SpotShape(Enum):
    circle = auto()
    rectangle = auto()

    def __str__(self):
        return str(self.name)


class Graph(BackableObject, BoundingBoxable):
    def __init__(
        self,
        untransformed_node_positions: Optional[np.ndarray] = None,
        edge_indices: Optional[np.ndarray] = None,
        edge_features: Optional[np.ndarray] = None,
        undirected: Optional[bool] = None,
        obs: Optional[pd.DataFrame] = None,
        backing: Optional[Union[h5py.Group, h5py.Dataset]] = None,
    ):
        super().__init__(backing)
        self._parentdataset = None
        if backing is not None:
            if (
                untransformed_node_positions is not None
                or edge_indices is not None
                or edge_features is not None
                or undirected is not None
                or obs is not None
            ):
                raise ValueError("attempting to specify masks for a non-empty backing store")
            self._untransformed_node_positions = backing["untransformed_node_positions"]
            self._edge_indices = backing["edge_indices"]
            self._edge_features = backing["edge_features"]
            a = _get_hdf5_attribute(backing.attrs, "undirected")
            assert a is not None
            self._undirected = a
            self._obs = read_attribute(backing["obs"])
        else:
            assert untransformed_node_positions is not None and undirected is not None
            self.untransformed_node_positions = untransformed_node_positions
            self.undirected = undirected
            if edge_indices is not None:
                self.edge_indices = edge_indices
            else:
                self.edge_indices = np.array([[]])

            if edge_features is not None:
                self.edge_features = edge_features
            else:
                self.edge_features = np.array([[]])

            if obs is not None:
                self.obs = obs
            else:
                self.obs = pd.DataFrame()
        self.update_obs_from_nodes()

    @property
    def untransformed_node_positions(self):
        return self._untransformed_node_positions

    @untransformed_node_positions.setter
    def untransformed_node_positions(self, o):
        self._untransformed_node_positions = o
        self.obj_has_changed("untransformed_node_positions")

    @property
    def transformed_node_positions(self):
        return self._parentdataset.anchor.transform_coordinates(self.untransformed_node_positions)

    @property
    def edge_indices(self):
        return self._edge_indices

    @edge_indices.setter
    def edge_indices(self, o):
        self._edge_indices = o
        self.obj_has_changed("edge_indices")

    @property
    def edge_features(self):
        return self._edge_features

    @edge_features.setter
    def edge_features(self, o):
        self._edge_features = o
        self.obj_has_changed("edge_features")

    @property
    def undirected(self):
        return self._undirected

    @undirected.setter
    def undirected(self, o):
        self._undirected = o
        self.obj_has_changed("undirected")

    @property
    def anchor(self):
        return self._parentdataset.anchor

    @anchor.setter
    def anchor(self, new_anchor):
        self._parentdataset.anchor = new_anchor

    @property
    def coordinate_unit(self):
        return self._parentdataset.coordinate_unit

    @coordinate_unit.setter
    def coordinate_unit(self, o):
        self._parentdataset.coordinate_unit = o

    def __len__(self):
        return len(self.untransformed_node_positions)

    def update_obs_from_nodes(self):
        # if the dataframe is not empty
        if len(self.obs.columns) != 0:
            raise ValueError(
                "replacing the old obs is only performed when obs is an empty DataFrame or it is None"
            )
        assert len(self.untransformed_node_positions) > 0
        df = pd.DataFrame(index=range(len(self.untransformed_node_positions)))
        self.obs = df

    @property
    def obs(self) -> pd.DataFrame:
        return self._obs

    @obs.setter
    def obs(self, new_obs):
        self._obs = new_obs
        self.obj_has_changed("obs")

    @property
    def n_obs(self) -> int:
        return self._obs.shape[0]

    @property
    def transformed_node_positions(self):
        return self.anchor.transform_coordinates(self.untransformed_node_positions[...])

    def _write_impl(self, obj: Union[h5py.Group, h5py.Dataset]):
        super()._write_impl(obj)
        if self.has_obj_changed("untransformed_node_positions"):
            if "untransformed_node_positions" in obj:
                del obj["untransformed_node_positions"]
            obj.create_dataset(
                "untransformed_node_positions", data=self.untransformed_node_positions
            )
        if self.has_obj_changed("edge_indices"):
            if "edge_indices" in obj:
                del obj["edge_indices"]
            obj.create_dataset("edge_indices", data=self.edge_indices)
        if self.has_obj_changed("edge_features"):
            if "edge_features" in obj:
                del obj["edge_features"]
            obj.create_dataset("edge_features", data=self.edge_features)

    def _write_attributes_impl(self, obj: Union[h5py.Dataset, h5py.Group]):
        super()._write_attributes_impl(obj)
        if self.has_obj_changed("undirected"):
            obj.attrs['undirected'] = self.undirected
        if self.has_obj_changed("obs"):
            write_attribute(
                obj,
                "obs",
                self.obs,
                dataset_kwargs={"compression": "gzip", "compression_opts": 9},
            )

    def __repr__(self):
        repr_str = f"│   ├── (Graph) with {self.n_obs} obs: {', '.join(self.obs)}"
        repr_str = "│   └──".join(repr_str.rsplit("│   ├──", 1))
        return repr_str

    @property
    def ndim(self):
        return self.untransformed_node_positions.shape[1]

    @property
    def _untransformed_bounding_box(self) -> BoundingBox:
        ##
        assert self.ndim == 2
        x_min, y_min = np.min(self.untransformed_node_positions, axis=0)
        x_max, y_max = np.max(self.untransformed_node_positions, axis=0)
        bb = BoundingBox(x0=x_min, x1=x_max, y0=y_min, y1=y_max)
        return bb

    @staticmethod
    def _encodingtype():
        return "graph"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def crop(self, bounding_box: BoundingBox):
        NotImplementedError()

    def subset_obs(self, indices: np.array, inplace: bool = False):
        if not inplace:
            untransformed_node_positions = self.untransformed_node_positions
            untransformed_node_positions = untransformed_node_positions[indices, :]

            obs = self.obs.iloc[indices, :].copy()

            node_mask = np.zeros(len(self), dtype=bool)
            node_mask[indices] = True
            edge_mask = node_mask[self.edge_indices[:, 0]] & node_mask[self.edge_indices[:, 1]]
            edge_indices = self.edge_indices[edge_mask, :]
            edge_features = self.edge_features[edge_mask, :]
            # relabel
            assert np.alltrue(np.cumsum(node_mask)[indices] - 1 == np.arange(len(indices)))
            edge_indices = np.cumsum(node_mask)[edge_indices] - 1

            g = Graph(
                untransformed_node_positions=untransformed_node_positions,
                edge_indices=edge_indices,
                edge_features=edge_features,
                undirected=self.undirected,
                obs=obs,
            )
            return g
        else:
            raise NotImplementedError()

    # partially inspired by https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert
    # .html
    def to_networkx(self) -> Tuple[Union[nx.Graph, nx.DiGraph], Dict[int, np.array]]:
        """
        warning: as a limitaiton of networkx, the order of edges is not guaranteed to be the same of the one of self.edge_indices
        :return:
        """
        if self.undirected:
            g = nx.Graph()
        else:
            g = nx.DiGraph()

        g.add_nodes_from(range(self.__len__()))

        edges = [(u, v, w) for ((u, v), w) in zip(self.edge_indices, self.edge_features)]
        g.add_weighted_edges_from(edges)

        if self._parentdataset is not None:
            positions = {i: pos for i, pos in enumerate(self.transformed_node_positions)}
        else:
            positions = {i: pos for i, pos in enumerate(self.untransformed_node_positions)}
        return g, positions

    # flake8: noqa: C901
    def plot(
        self,
        node_colors: ColorsType = "black",
        outline_colors: ColorsType = None,
        edge_colors: ColorsType = "black",
        edge_feature_index: Optional[int] = None,
        edge_cmap=None,
        node_size: int = 5,
        edge_size: int = 1,
        ax: matplotlib.axes.Axes = None,
        alpha: float = 1.0,
        show_title: bool = True,
        show_legend: bool = True,
        show_colorbar: bool = True,
        bounding_box: Optional[BoundingBox] = None,
        figsize: Optional[Tuple[int]] = None,
        categories_colors: Optional[Dict[str, ColorType]] = None,
    ):
        if edge_cmap is not None:
            edge_colors = None

        if edge_colors is not None:
            assert edge_feature_index is None
        else:
            if (
                len(self.edge_features.shape) == 1
                or len(self.edge_features.shape) == 2
                and self.edge_features.shape[1] == 1
            ):
                assert edge_feature_index is None or edge_feature_index == 0
            else:
                assert len(self.edge_features.shape) == 2
                assert edge_feature_index is not None and isinstance(edge_feature_index, int)
                assert edge_feature_index >= 0 and edge_feature_index < self.edge_features.shape[1]
            assert edge_cmap is not None
        if bounding_box is not None:
            raise NotImplementedError()
        if outline_colors is not None:
            raise NotImplementedError()

        n = len(self.obs)

        fill_color_array, plotting_a_category, title, _legend = handle_categorical_plot(
            node_colors, obs=self.obs, categories_colors=categories_colors
        )
        if not plotting_a_category:
            fill_color_array = get_color_array_rgba(node_colors, n)

        outline_color_array, plotting_a_category, title, _legend = handle_categorical_plot(
            outline_colors, obs=self.obs, categories_colors=categories_colors
        )
        if not plotting_a_category:
            outline_color_array = get_color_array_rgba(outline_colors, n)

        g, pos = self.to_networkx()
        colorbar_scalar_mappable = None
        if edge_colors is not None:
            edge_color_array = get_color_array_rgba(edge_colors, len(self.edge_indices))
        else:
            assert edge_cmap is not None
            ll = list(g.edges.values())
            ll = [x["weight"].reshape(1, -1) for x in ll]
            edge_features = np.concatenate(ll, axis=0)
            assert len(edge_features.shape) == 2
            if edge_features[1] == 1:
                edge_values = edge_features.flatten()
            else:
                edge_values = edge_features[:, edge_feature_index].flatten()
            assert len(edge_values.shape) == 1
            edge_vmax = np.max(edge_values)
            edge_vmin = np.min(edge_values)
            normalized = (edge_values - edge_vmin) / (edge_vmax - edge_vmin)
            edge_color_array = np.array([edge_cmap(c) for c in normalized])
            if show_colorbar:
                cnorm = matplotlib.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
                colorbar_scalar_mappable = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=edge_cmap)

        for a in [fill_color_array, outline_color_array, edge_color_array]:
            apply_alpha(a, alpha=alpha)

        ##
        if ax is not None and figsize is not None:
            raise ValueError("ax and figsize cannot be both non-None")

        if ax is None:
            plt.figure(figsize=figsize)
            axs = plt.gca()
        else:
            axs = ax

        if colorbar_scalar_mappable is not None:
            plt.colorbar(
                colorbar_scalar_mappable,
                orientation="horizontal",
                location="bottom",
                ax=axs,
                shrink=0.6,
                pad=0.1,
            )

        # kwargs = {}
        # if edge_values is not None:
        #     kwargs["edge_vmin"] = np.min(edge_values)
        #     kwargs["edge_vmax"] = np.max(edge_values)

        nx.drawing.nx_pylab.draw_networkx(
            g,
            pos=pos,
            # edgelist=edges_to_plot,
            node_size=node_size,
            # linewidths=,  # for the outline color
            with_labels=False,
            width=edge_size,
            arrows=False,
            node_color=fill_color_array,
            edge_color=edge_color_array,
            ax=axs,
            # **kwargs,
        )
        # restore proper margins and ticks that are modified by draw_networkx
        # NOTE:
        # actually this functions creates some tiny margings that are larger than what we would get by plotting just,
        # sy, a visium regions object, but it is just a small graphical difference that we accept
        axs.margins(0)
        axs.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=True,
            labelbottom=True,
            labelleft=True,
        )

        if plotting_a_category:
            if show_title:
                axs.set_title(title)
            if show_legend:
                axs.legend(
                    handles=_legend,
                    frameon=False,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.1),
                    ncol=len(_legend),
                )
        # when the plot is invoked with my_fov.masks.plot(), then self._parentdataset._adjust_plot_lims() is called
        # once, when the plot is invoked with my_fov.plot(), then self._parentdataset._adjust_plot_lims() is called
        # twice (one now and one in the code for plotting FieldOfView subclasses). This should not pose a problem
        if self._parentdataset is not None:
            self._parentdataset._adjust_plot_lims(ax=axs)
        if ax is None:
            plt.show()
        ##

    @classmethod
    def disconnected_graph(cls, untransformed_node_positions: np.ndarray, undirected: bool = True):
        g = Graph(
            untransformed_node_positions=untransformed_node_positions,
            undirected=undirected,
        )
        return g

    def clear_edges(self):
        self.edge_indices = np.array([[]])
        self.edge_features = np.array([[]])
        self.commit_changes_on_disk()

    def compute_knn_edges(self, k: int = 10, max_distance: Optional[float] = None):
        if self.edge_indices.size > 0 or self.edge_features.size > 0:
            raise RuntimeError(
                "the graph already contains edges, call .clear_edges() to remove them and then call "
                "this function again"
            )
        if k <= 1:
            raise ValueError("expected k >= 2 (not considering self-loops)")
        kk = min(k, len(self))
        centers = self.untransformed_node_positions[...]
        neighbors = NearestNeighbors(n_neighbors=kk, algorithm="ball_tree").fit(centers)
        distances, indices = neighbors.kneighbors(centers)
        assert np.all(np.diff(distances, axis=1) >= 0)
        # check that every element is a knn of itself, and this is the first neighbor
        assert all(indices[:, 0] == np.array(range(len(self))))
        edges = []
        weights = []
        # extra_data = []
        # extra_data.append(indices[:, :GRAPH_KNN_K])
        added = set()
        for i in range(indices.shape[0]):
            for j in range(1, indices.shape[1]):
                edge = (i, indices[i, j])
                if edge not in added:
                    added.add(edge)
                    added.add((edge[1], edge[0]))
                    d = distances[i, j]
                    weight = d
                    if max_distance is None or d < max_distance:
                        edges.append(edge)
                        weights.append(weight)
        self.edge_indices = np.array(edges)
        if len(weights) == 0:
            print('no edge added, consider increasing the value of max_distance')
        self.edge_features = np.array(weights).reshape((len(self.edge_indices), -1))
        self.commit_changes_on_disk()

    def compute_rbfk_edges(self, length_scale: float, max_distance: Optional[float] = None):
        edges = []
        weights = []
        centers = self.untransformed_node_positions[...]
        tree = cKDTree(centers)
        d = 2 * length_scale ** 2
        for i in range(len(self)):
            a = np.array(centers[i])
            weight_threshold = np.exp(-(max_distance ** 2) / d)
            distance_threshold = math.sqrt(0 - math.log(weight_threshold) * d)
            neighbors = tree.query_ball_point(a, distance_threshold, p=2)
            for j in neighbors:
                if i >= j:
                    continue
                b = np.array(centers[j])
                c = a - b
                weight = np.exp(-np.dot(c, c) / d)
                assert weight >= weight_threshold, (weight, weight_threshold)
                edges.append([i, j])
                weights.append(weight)
        self.edge_indices = np.array(edges)
        self.edge_features = np.array(weights).reshape((len(self.edge_indices), -1))
        self.commit_changes_on_disk()

    def compute_proximity_edges(self, max_distance: float):
        edges = []
        weights = []
        centers = self.untransformed_node_positions[...]
        tree = cKDTree(centers)
        for i in range(len(self)):
            a = np.array(centers[i])
            neighbors = tree.query_ball_point(a, max_distance, p=2)
            for j in neighbors:
                if i >= j:
                    continue
                b = np.array(centers[j])
                c = a - b
                distance = np.sqrt(np.dot(c, c))
                assert distance <= max_distance, (distance, max_distance)
                edges.append([i, j])
                weights.append(distance)
        self.edge_indices = np.array(edges)
        self.edge_features = np.array(weights).reshape((len(self.edge_indices), -1))
        self.commit_changes_on_disk()

    def subgraph_of_neighbors(
        self,
        node_indices: Union[List[int] | int],
        subset_method: Literal["proximity", "knn"] = "knn",
        max_distance: Optional[float] = None,
        k: Optional[int] = None
    ):
        flatten_the_return = False
        if isinstance(node_indices, int):
            flatten_the_return = True
            node_indices = [node_indices]
        all_centers = self.untransformed_node_positions
        centers = all_centers[node_indices, :]
        is_nears = []
        if subset_method == "proximity":
            assert max_distance is not None
            assert k is None
            for center in centers:
                is_near = np.linalg.norm(all_centers - center, axis=1) < max_distance
                is_nears.append(is_near)
        elif subset_method == "knn":
            assert max_distance is None
            # raise NotImplementedError()
            g, positions = self.to_networkx()
            for node_index in node_indices:
                neighbors = list(g.neighbors(node_index))
                l = []
                for node in neighbors:
                    w = g.get_edge_data(node_index, node)["weight"]
                    l.append((node, w))
                l = sorted(l, key=lambda x: x[1])
                if k is None:
                    nearest = [l[i][0] for i in range(len(l))]
                else:
                    k = min(k, len(l))
                    nearest = [l[i][0] for i in range(k - 1)]
                nearest.append(node_index)
                is_near = np.zeros(len(all_centers), dtype=np.bool)
                is_near[np.array(nearest, dtype=np.long)] = True
                ##
                if False:
                    plt.figure()
                    ax = plt.gca()
                    ax.scatter(self.untransformed_node_positions[:, 0], self.untransformed_node_positions[:, 1], c='w', s=1)
                    for i in range(len(self)):
                        x, y = self.untransformed_node_positions[i, :]
                        c = 'w' if i not in nearest else 'r'
                        ax.annotate(str(i), (x, y), color=c)
                    plt.show()
                ##
                is_nears.append(is_near)
        else:
            raise ValueError()

        subgraphs = []
        center_indices = []
        nodes_to_keeps = []
        for is_near, node_index in zip(is_nears, node_indices):
            nodes_to_keep = np.where(is_near)[0]
            subgraph = self.subset_obs(indices=nodes_to_keep, inplace=False)
            center_index = (np.cumsum(is_near) - 1)[node_index]

            subgraphs.append(subgraph)
            center_indices.append(center_index)
            nodes_to_keeps.append(nodes_to_keep)
        if flatten_the_return:
            return subgraphs[0], center_indices[0], nodes_to_keeps[0]
        else:
            return subgraphs, center_indices, nodes_to_keeps
