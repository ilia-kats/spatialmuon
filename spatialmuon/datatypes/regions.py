from __future__ import annotations

from typing import Optional, Union, Literal, Callable, List, Dict, Tuple
from enum import Enum, auto
import warnings

import anndata._core.sparse_dataset
import numpy as np
import scipy.sparse.csr
from scipy.sparse import spmatrix
import pandas as pd
from shapely.geometry import Point, Polygon
from trimesh import Trimesh
import h5py
import matplotlib
import matplotlib.cm
import matplotlib.axes
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import scipy
from anndata import AnnData
from anndata._io.utils import read_attribute, write_attribute
from anndata._core.sparse_dataset import SparseDataset
import spatialmuon
from spatialmuon._core.masks import Masks
from spatialmuon._core.graph import Graph
from spatialmuon._core.anchor import Anchor
from spatialmuon._core.bounding_box import BoundingBox
from spatialmuon.datatypes.datatypes_utils import (
    regions_raster_plot,
    get_channel_index_from_channel_name,
    PlottingMethod,
    regions_raster_subset_var,
)
from tqdm.auto import tqdm
import warnings
import anndata

from spatialmuon import FieldOfView, SpatialIndex
from spatialmuon.utils import _read_hdf5_attribute, preprocess_3d_polygon_mask, ColorType


class Regions(FieldOfView):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        *,
        X: Optional[Union[np.ndarray, spmatrix]] = None,
        masks: Optional[Masks] = None,
        graph: Optional[Graph] = None,
        **kwargs,
    ):
        ndim = self.compute_ndim(X=X, masks=masks, graph=graph)
        kwargs["anchor"] = self.update_n_dim_in_anchor(ndim=ndim, backing=backing, **kwargs)
        super().__init__(backing, **kwargs)
        if backing is not None:
            if X is not None or masks is not None or graph is not None:
                raise ValueError("attempting to specify properties for a non-empty backing store")
            # self._index = SpatialIndex(
            #     backing=backing["index"], dimension=backing["coordinates"].shape[1], **index_kwargs
            # )
            assert "masks" in backing or "graph" in backing
            if "masks" in backing:
                masks = Masks(backing=backing["masks"])
                masks._parentdataset = self
                self._set_kv("masks", masks)
            if "graph" in backing:
                graph = Graph(backing=backing["graph"])
                graph._parentdataset = self
                self._set_kv("graph", graph)

            self._X = backing["X"]
        else:
            if masks is None and graph is None:
                raise ValueError('at least one between "masks" and "graph" need to be specify')
            if masks is not None:
                masks._parentdataset = self
                self.masks = masks
            if graph is not None:
                graph._parentdataset = self
                self.graph = graph

            if X is not None:
                self.X = X
            else:
                if masks is not None:
                    self.X = np.zeros((len(masks), 0))
                else:
                    assert graph is not None
                    self.X = np.zeros((len(graph), 0))

            # self._index = SpatialIndex(coordinates=self._coordinates, **index_kwargs)

        # we don't want to validate stuff coming from HDF5, this may break I/O
        # but mostly we can't validate for a half-initalized object
        # self.masks.validatefun = self.__validate_mask

    # @staticmethod
    # def __validate_mask(fov, key, mask):
    #     if mask.ndim is not None and mask.ndim != fov.ndim:
    #         return f"mask with {mask.ndim} dimensions is being added to field of view with {fov.ndim} dimensions"
    #     mask._parentdataset = fov
    #     return None
    @staticmethod
    def compute_ndim(X, masks, graph):
        ndim_x = None
        ndim_masks = None
        ndim_graph = None
        # self._X = None
        if X is not None:
            ndim_x = len(X.shape)
        if masks is not None:
            ndim_masks = masks.ndim
        if graph is not None:
            ndim_graph = graph.ndim
        ndims = {ndim_x, ndim_masks, ndim_graph}
        if None in ndims:
            ndims.remove(None)
        if len(ndims) == 0:
            return None
        elif len(ndims) == 1:
            return ndims.__iter__().__next__()
        else:
            raise ValueError("multiple values for ndim found")

    @property
    def _untransformed_bounding_box(self) -> BoundingBox:
        if self.masks is not None:
            return self.masks._untransformed_bounding_box
        else:
            assert self.graph is not None
            return self.graph._untransformed_bounding_box

    @property
    def X(self) -> Union[np.ndarray, spmatrix, h5py.Dataset, SparseDataset]:
        if self.is_backed:
            if isinstance(self._X, h5py.Group):
                return SparseDataset(self._X)
            else:
                return self._X
        else:
            return self._X

    @X.setter
    def X(self, new_X):
        self._X = new_X
        self.obj_has_changed("X")

    @property
    def masks(self):
        if "masks" in self:
            return self["masks"]
        else:
            return None

    @masks.setter
    def masks(self, new_masks):
        new_masks._parentdataset = self
        self["masks"] = new_masks

    @property
    def graph(self):
        if "graph" in self:
            return self["graph"]
        else:
            return None

    @graph.setter
    def graph(self, new_graph):
        new_graph._parentdataset = self
        self["graph"] = new_graph

    @property
    def obs(self) -> pd.DataFrame:
        if self.masks is not None:
            return self.masks.obs
        else:
            assert self.graph is not None
            return self.graph.obs

    @obs.setter
    def obs(self, new_obs):
        if self.masks is not None:
            self.masks.obs = new_obs
        else:
            assert self.graph is not None
            self.graph.obs = new_obs

    # def _getitem(
    #     self,
    #     mask: Optional[Union[Polygon, Trimesh]] = None,
    #     genes: Optional[Union[str, list[str]]] = None,
    #     polygon_method: Literal["project", "discard"] = "discard",
    # ) -> AnnData:
    #     raise NoImplementedError()
    # if mask is not None:
    #     if self.ndim == 2:
    #         if not isinstance(mask, Polygon):
    #             raise TypeError("Only polygon masks can be applied to 2D FOVs")
    #         idx = sorted(self._index.intersection(mask.bounds))
    #         obs = self.obs.iloc[idx, :].intersection(mask)
    #         X = self.X[idx, :][~obs.is_empty, :]
    #         obs = self.obs[~obs.is_empty]
    #         coords = np.vstack(obs.geometry)
    #         obs.drop(obs.geometry.name, axis=1, inplace=True)
    #     else:
    #         if isinstance(mask, Polygon):
    #             bounds = preprocess_3d_polygon_mask(mask, self._coordinates, polygon_method)
    #             idx = sorted(self._index.intersection(bounds))
    #             sub = self._obs.iloc[idx, :].intersection(mask)
    #             nemptyidx = ~sub.is_empty
    #         elif isinstance(mask, Trimesh):
    #             idx = sorted(self._index.intersection(mask.bounds.reshape(-1)))
    #             sub = self._obs.iloc[idx, :]
    #             nemptyidx = mask.contains(np.vstack(sub.geometry))
    #         else:
    #             raise TypeError("unknown masks type")
    #         X = (self.X[idx, :][nemptyidx, :],)
    #         obs = sub.iloc[nemptyidx, :]
    # else:
    #     X = self.X[()]
    #     obs = self.obs
    # ad = AnnData(X=X, obs=obs, var=self.var)
    # if genes is not None:
    #     ad = ad[:, genes]
    # return ad

    @staticmethod
    def _encodingtype() -> str:
        return "fov-array"

    @staticmethod
    def _encodingversion() -> str:
        return "0.1.0"

    # @property
    # def _backed_children(self) -> Dict["BackableObject"]:
    #     return {'masks': self.masks}

    # def _set_backing(self, grp: Optional[h5py.Group]):
    #     super()._set_backing(grp)
    #     if grp is not None:
    #         assert isinstance(grp, h5py.Group)
    #         # self._backing should be reassigned from one of the caller functions (set_backing from BackableObject),
    #         # but to be safe let's set it to None explicity here
    #         self._backing = None
    #         self._write_impl(grp)
    #         # self.masks.set_backing(grp, "masks")
    #         # self._X = None
    #         # self._index.set_backing(grp, "index")
    #     else:
    #         print("who is calling me?")
    #         assert self.is_backed
    #         # self._X = read_attribute(grp, "X")
    #         # self._index.set_backing(None)
    #         # self.masks.set_backing(None)

    def _write_impl(self, grp: h5py.Group):
        super()._write_impl(grp)
        if self.compressed_storage:
            if self.has_obj_changed("X"):
                write_attribute(
                    grp, "X", self.X, dataset_kwargs={"compression": "gzip", "compression_opts": 9}
                )
        else:
            if self.has_obj_changed("X"):
                # for debugging (waaay faster, but more space on disk), for the future let's maybe make the compression
                # optional
                write_attribute(grp, "X", self.X)

        # self._index._write(grp, "index")

    def _write_attributes_impl(self, obj):
        super()._write_attributes_impl(obj)
        pass

    # TODO: this function shares code with Raster._plot_in_grid(), unify better at some point putting for instance
    #  the shared code in datatypes_utils.py
    def _plot_in_grid(
        self,
        channels_to_plot: list[str],
        grid_size: Union[int, list[int]] = 1,
        preprocessing: Optional[Callable] = None,
        cmap: Union[
            matplotlib.colors.Colormap, list[matplotlib.colors.Colormap]
        ] = matplotlib.cm.viridis,
        suptitle: Optional[str] = None,
        alpha: float = 1.0,
    ):
        idx = get_channel_index_from_channel_name(self.var, channels_to_plot[0])
        # TODO: get this info by calling a get_bounding_box() function, which shuold take into account for alignment
        #  information
        # (x, y) = self.X[:, :, idx].shape
        # cell_size_x = 2 * x / max(x, y)
        # cell_size_y = 2 * y / max(x, y)
        cell_size_x = cell_size_y = 3
        fig, axs = plt.subplots(
            grid_size[1],
            grid_size[0],
            figsize=(cell_size_y * grid_size[1], cell_size_x * grid_size[0]),
        )
        if len(channels_to_plot) > 1:
            axs = axs.flatten()

        for idx, channel in enumerate(tqdm(channels_to_plot, disable=len(channels_to_plot) < 20)):
            self.plot(
                channels=channel,
                preprocessing=preprocessing,
                cmap=cmap,
                ax=axs[idx],
                show_legend=False,
                show_colorbar=False,
                show_scalebar=idx == 0,
                alpha=alpha,
            )
        for idx in range(len(channels_to_plot), grid_size[0] * grid_size[1]):
            axs[idx].set_axis_off()
        if suptitle is not None:
            plt.suptitle(suptitle)
        plt.subplots_adjust()
        plt.tight_layout()
        plt.show()

    # TODO: code very similar to Raster._plot_in_canvas(), maybe unify
    def _plot_in_canvas(
        self,
        channels_to_plot: list[str],
        rgba: bool,
        preprocessing: Optional[Callable] = None,
        cmap: Union[
            matplotlib.colors.Colormap, list[matplotlib.colors.Colormap]
        ] = matplotlib.cm.viridis,
        ax: matplotlib.axes.Axes = None,
        alpha: float = 1.0,
        bounding_box: Optional[BoundingBox] = None,
        fill_color: Optional[Union[Literal["channel"], ColorType]] = "channel",
        outline_color: Optional[Union[Literal["channel"], ColorType]] = None,
    ):
        if rgba:
            indices = [
                get_channel_index_from_channel_name(self.var, channel)
                for channel in channels_to_plot
            ]
            indices = np.array(indices)
            data_to_plot = self.X[:, indices]

            x = data_to_plot if preprocessing is None else preprocessing(data_to_plot)
            if np.min(x) < 0.0 or np.max(x) > 1.0:
                warnings.warn(
                    "the data is not in the [0, 1] range. Plotting the result of an affine transformation to "
                    "make the data in [0, 1]."
                )
                old_shape = x.shape
                x = np.reshape(x, (-1, old_shape[-1]))
                a = np.min(x, axis=0)
                b = np.max(x, axis=0)
                x = (x - a) / (b - a)
                x = np.reshape(x, old_shape)
            colors = x
            a = 1.0
        else:
            for idx, channel in enumerate(channels_to_plot):
                a = 1 / (max(len(channels_to_plot) - 1, 2)) if idx > 0 else 1
                channel_index = get_channel_index_from_channel_name(self.var, channel)
                data_to_plot = self.X[:, channel_index]
                x = data_to_plot if preprocessing is None else preprocessing(data_to_plot)
                assert len(cmap) == 1
                cmap = cmap[0]
                cnorm = matplotlib.colors.Normalize(vmin=np.min(x), vmax=np.max(x))
                sm = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmap)

                # instead of calling imshow passing cnorm and cmap we are using the plotting function defined for
                # masks and passing already the appropriate colors. We compute the colors manually
                def normalizer(e):
                    d = e.max() - e.min()
                    if np.isclose(d, 0):
                        return np.zeros_like(e)
                    else:
                        return (e - e.min()) / d

                if isinstance(x, scipy.sparse.csr.csr_matrix):
                    z = x.todense()
                else:
                    z = x
                if type(z) == np.ndarray:
                    z = z.flatten()
                elif type(z) == np.matrix:
                    z = z.A1
                z = normalizer(z)
                colors = cmap(z)
        fill_colors = colors if fill_color == "channel" else fill_color
        outline_colors = colors if outline_color == "channel" else outline_color
        if self.masks is not None:
            self.masks.plot(
                fill_colors=fill_colors,
                outline_colors=outline_colors,
                ax=ax,
                alpha=a * alpha,
                bounding_box=bounding_box,
            )
        else:
            assert self.graph is not None
            print("plot somthing")
            raise NotImplementedError()
        # im = ax.imshow(x, cmap=cmap[idx], alpha=a)
        # code explained in raster.py
        if len(channels_to_plot) == 1:
            return sm
        else:
            return None

    def plot(
        self,
        channels: Optional[Union[str, list[str], int, list[int]]] = "all",
        fill_color: Optional[Union[Literal["channel"], ColorType]] = "channel",
        outline_color: Optional[Union[Literal["channel"], ColorType]] = None,
        grid_size: Union[int, list[int]] = 1,
        method: PlottingMethod = "auto",
        preprocessing: Optional[Callable] = None,
        cmap: Union[
            matplotlib.colors.Colormap, list[matplotlib.colors.Colormap]
        ] = matplotlib.cm.viridis,
        ax: matplotlib.axes.Axes = None,
        show_title: bool = True,
        show_legend: bool = True,
        show_colorbar: bool = True,
        show_scalebar: bool = True,
        suptitle: Optional[str] = None,
        alpha: float = 1,
        bounding_box: Optional[BoundingBox] = None,
        figsize: Optional[Tuple[int]] = None
    ):
        if self.var is None or len(self.var.columns) == 0:
            if self.masks is not None:
                warnings.warn(
                    "No quantities to plot, plotting the masks with random colors instead. For more options in plotting masks use .masks.plot()"
                )
                self.masks.plot(ax=ax, figsize=figsize)
            else:
                assert self.graph is not None
                warnings.warn(
                    "No quantities to plot, plotting the graph. For more options in "
                    "plotting the graph use .graph.plot()"
                )
                self.graph.plot(ax=ax, figsize=figsize)
        else:
            regions_raster_plot(
                self,
                channels=channels,
                fill_color=fill_color,
                outline_color=outline_color,
                grid_size=grid_size,
                preprocessing=preprocessing,
                method=method,
                cmap=cmap,
                ax=ax,
                show_title=show_title,
                show_legend=show_legend,
                show_colorbar=show_colorbar,
                show_scalebar=show_scalebar,
                suptitle=suptitle,
                alpha=alpha,
                bounding_box=bounding_box,
                figsize=figsize
            )
        # the approach described below is quite convoluted, it is not needed now, but be aware that if it is needed
        # this is a way to proceed
        # sometimes a copy of a masks object is created and used for operations, for instance when we plot into a
        # boundingbox we created a new masks object and we subset the obs. Still, the new masks points to the old
        # Regions object (self), and sometimes calls functions on it that access the .masks slot, for instance
        # self.bounding_box is accessing self.masks, so the old masks!
        # I think that is not needed to have an access to the new masks, but if it is, we can replace the old
        # self.masks with "new_masks" in the place in which the new masks object is created and then here we can set
        # it back to self.
        # In doing so we must not call self.masks = new_masks or self.masks = self since this is triggering the
        # backing system. We should instead call super(self, UserDict).__setitem('masks', new_masks)
        # the restoring back self should be done via old_masks = self.masks in the beggining of this function, and
        # self.super(self, UserDict).__setitem__('masks', old_masks) at the end of this function

    def __repr__(self):
        repr_str = f"(Regions) with {self.n_var} var\n"
        if self.masks is not None:
            repr_str += str(self.masks)
        if self.graph is not None:
            repr_str += str(self.graph)
        return repr_str

    def crop(self, bounding_box: BoundingBox):
        raise NotImplementedError()

    @property
    def untransformed_centers(self):
        assert self.masks is not None or self.graph is not None
        if self.masks is not None:
            return self.masks.untransformed_masks_centers
        else:
            return self.graph.untransformed_node_positions

    @property
    def transformed_centers(self):
        assert self.masks is not None or self.graph is not None
        if self.masks is not None:
            return self.masks.transformed_masks_centers
        else:
            return self.graph.transformed_node_positions

    def compute_knn_graph(self, k: int = 10, max_distance_in_units: Optional[float] = None):
        if max_distance_in_units is not None:
            max_distance = self.anchor.inverse_transform_length(max_distance_in_units)
        else:
            max_distance = max_distance_in_units
        g = Graph.disconnected_graph(untransformed_node_positions=self.untransformed_centers)
        g.compute_knn_edges(k=k, max_distance=max_distance)
        g._parentdataset = self
        return g

    def compute_rbfk_graph(
        self, length_scale_in_units: float, max_distance_in_units: Optional[float] = None
    ):
        length_scale = self.anchor.inverse_transform_length(length_scale_in_units)
        if max_distance_in_units is not None:
            max_distance = self.anchor.inverse_transform_length(max_distance_in_units)
        else:
            max_distance = max_distance_in_units
        g = Graph.disconnected_graph(untransformed_node_positions=self.untransformed_centers)
        g.compute_rbfk_edges(length_scale=length_scale, max_distance=max_distance)
        g._parentdataset = self
        return g

    def compute_proximity_graph(self, max_distance_in_units: float):
        max_distance = self.anchor.inverse_transform_length(max_distance_in_units)
        g = Graph.disconnected_graph(untransformed_node_positions=self.untransformed_centers)
        g.compute_proximity_edges(max_distance=max_distance)
        g._parentdataset = self
        return g

    def subset_obs(self, indices: np.array, inplace: bool = False):
        def subset(instance: Regions):
            instance.X = instance.X[indices, ...]
            if self.masks is not None:
                self.masks.subset_obs(indices=indices, inplace=True)
            if self.graph is not None:
                self.graph.subset_obs(indices=indices, inplace=True)
            instance.commit_changes_on_disk()
        if inplace:
            subset(self)
        else:
            o = self.clone()
            subset(o)
            return o

    def subset_var(self, indices: np.array, inplace: bool = False):
        regions_raster_subset_var(instance=self, indices=indices, inplace=inplace)
