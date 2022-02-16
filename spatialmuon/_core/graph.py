from __future__ import annotations

from collections.abc import MutableMapping
from typing import Optional, Union, Literal, TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.axes
import matplotlib.cm
import matplotlib.patches
import matplotlib.collections
import numpy as np
import h5py
import networkx as nx
import copy

import shapely.geometry
from shapely.geometry import Polygon
from trimesh import Trimesh
from skimage.measure import find_contours
from anndata._io.utils import read_attribute, write_attribute
import pandas as pd
import skimage.measure
import vigra
from functools import cached_property
from enum import Enum, auto

from spatialmuon._core.backing import BackableObject
from spatialmuon.utils import (
    _read_hdf5_attribute,
    UnknownEncodingException,
    _get_hdf5_attribute,
    ColorsType,
    handle_categorical_plot,
    get_color_array_rgba,
    normalize_color,
apply_alpha
)
from spatialmuon._core.bounding_box import BoundingBoxable, BoundingBox

if TYPE_CHECKING:
    from spatialmuon import Raster, Regions


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
            self._undirected = a
            self._obs = read_attribute(backing["obs"])
        else:
            assert (
                untransformed_node_positions is not None
                and edge_indices is not None
                and undirected is not None
            )
            self.untransformed_node_positions = untransformed_node_positions
            self.edge_indices = edge_indices
            self.undirected = undirected

            if edge_features is not None:
                self.edge_features = edge_features
            else:
                self.edge_features = np.zeros([[]])

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
            write_attribute(
                obj,
                "undirected",
                self.undirected,
            )
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

    def subset_obs(self, indices: np.array):
        NotImplementedError()

    # partially inspired by https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert
    # .html
    def to_networkx(self):
        if self.undirected:
            g = nx.Graph()
        else:
            g = nx.DiGraph()

        g.add_nodes_from(range(self.__len__()))

        for i, (u, v) in enumerate(self.edge_indices):
            g.add_edge(u, v)

        print("TODO: convert also attributes")
        positions = {i: pos for i, pos in enumerate(self.transformed_node_positions)}
        return g, positions

    # flake8: noqa: C901
    def plot(
        self,
        node_colors: ColorsType = "black",
        outline_colors: ColorsType = None,
        edge_colors: ColorsType = "black",
        edge_values: Optional[np.array] = None,
        edge_cmap=None,
        edge_vmin: Optional[float] = None,
        edge_vmax: Optional[float] = None,
        node_size: int = 5,
        edge_size: int = 1,
        ax: matplotlib.axes.Axes = None,
        alpha: float = 1.0,
        show_title: bool = True,
        show_legend: bool = True,
        bounding_box: Optional[BoundingBox] = None,
    ):
        if edge_colors is not None:
            assert edge_values is None
            assert edge_cmap is None
            assert edge_vmin is None
            assert edge_vmax is None
        else:
            assert edge_values is not None
            assert edge_cmap is not None
        if bounding_box is not None:
            raise NotImplementedError()
        if outline_colors is not None:
            raise NotImplementedError()

        n = len(self.obs)

        fill_color_array, plotting_a_category, title, _legend = handle_categorical_plot(
            node_colors, obs=self.obs
        )
        if not plotting_a_category:
            fill_color_array = get_color_array_rgba(node_colors, n)

        outline_color_array, plotting_a_category, title, _legend = handle_categorical_plot(
            outline_colors, obs=self.obs
        )
        if not plotting_a_category:
            outline_color_array = get_color_array_rgba(outline_colors, n)

        if edge_colors is not None:
            edge_color_array = get_color_array_rgba(edge_colors, len(self.edge_indices))
        else:
            assert len(edge_values.shape) == 1
            if edge_vmax is None:
                edge_vmax = np.max(edge_values)
            if edge_vmin is None:
                edge_vmin = np.min(edge_values)
            normalized = (edge_values - edge_vmin) / (edge_vmax - edge_vmin) * 255.
            edge_color_array = np.array([edge_cmap(c) for c in normalized])

        for a in [fill_color_array, outline_color_array, edge_color_array]:
            apply_alpha(a, alpha=alpha)


        ##
        print('ehi')
        if ax is None:
            plt.figure()
            axs = plt.gca()
        else:
            axs = ax

        g, pos = self.to_networkx()
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
