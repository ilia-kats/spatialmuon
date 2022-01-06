from __future__ import annotations

from typing import Optional, Union, Literal, Callable
import warnings

import math
import numpy as np
import h5py
from shapely.geometry import Polygon, Point, MultiPoint
from trimesh import Trimesh
import matplotlib
import matplotlib.cm
import matplotlib.axes
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt

from spatialmuon import FieldOfView
from spatialmuon.utils import _get_hdf5_attribute
from spatialmuon.datatypes.datatypes_utils import (
    regions_raster_plot,
    get_channel_index_from_channel_name,
)
from spatialmuon._core.masks import Masks


class Raster(FieldOfView):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        *,
        X: Optional[np.ndarray] = None,
        px_dimensions: Optional[np.ndarray] = None,
        px_distance: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if backing is not None:
            self._ndim = backing["X"].ndim - 1
            if self._ndim == 1:
                self._ndim = 2
            self._px_distance = _get_hdf5_attribute(backing.attrs, "px_distance")
            self._px_dimensions = _get_hdf5_attribute(backing.attrs, "px_dimensions")
            self._X = None
        else:
            if X is None:
                raise ValueError("no data and no backing store given")
            self._X = X
            self._ndim = X.ndim if X.ndim == 2 else X.ndim - 1
            if self._ndim < 2 or self._ndim > 3:
                raise ValueError("image dimensionality not supported")
            self._px_dimensions = px_dimensions
            self._px_distance = px_distance
            if self._px_dimensions is not None:
                self._px_dimensions = np.asarray(self._px_dimensions).squeeze()
                if self._px_dimensions.shape[0] != self._ndim or self._px_dimensions.ndim != 1:
                    raise ValueError("pixel_size dimensionality is inconsistent with X")
            if self._px_distance is not None:
                self._px_distance = np.asarray(self._px_distance).squeeze()
                if self._px_distance.shape[0] != self._ndim or self._px_distance.ndim != 1:
                    raise ValueError("pixel_distance dimensionality is inconsistent with X")
        super().__init__(backing, **kwargs)

    @property
    def ndim(self):
        return self._ndim

    @property
    def X(self) -> Union[np.ndarray, h5py.Dataset]:
        if self.isbacked:
            return self.backing["X"]
        else:
            return self._X

    @property
    def px_dimensions(self) -> np.ndarray:
        if self._px_dimensions is None:
            return np.ones(self.ndim, np.uint8)
        else:
            return self._px_dimensions

    @property
    def px_distance(self) -> np.ndarray:
        if self._px_distance is None:
            return self.px_dimensions
        else:
            return self._px_distance

    # flake8: noqa: C901
    def _getitem(
        self,
        mask: Optional[Union[Polygon, Trimesh]] = None,
        genes: Optional[Union[str, list[str]]] = None,
        polygon_method: Literal["project", "discard"] = "discard",
    ):
        raise NotImplementedError()
        # if mask is not None:
        #     if self.ndim == 3:
        #         if isinstance(mask, Polygon):
        #             if polygon_method == "project":
        #                 warnings.warn(
        #                     "Method `project` not possible with raster FOVs. Using `discard`."
        #                 )
        #         elif isinstance(mask, Trimesh):
        #             lb, ub = np.floor(mask.bounds[0, :]), np.ceil(mask.bounds[1, :])
        #             data = self.X[lb[0] : ub[0], lb[1] : ub[1], lb[2] : ub[2], ...]
        #             coords = np.stack(
        #                 np.meshgrid(
        #                     range(ub[0] - lb[0] + 2),
        #                     range(ub[1] - lb[1] + 2),
        #                     range(ub[2] - lb[2] + 2),
        #                 ),
        #                 axis=1,
        #             )
        #             coords = coords[mask.contains(coords), :]
        #             return data[coords[:, 1], coords[:, 0], coords[:, 2], ...]
        #     if not isinstance(mask, Polygon):
        #         raise TypeError("Only polygon masks can be applied to 2D FOVs")
        #     bounds = mask.bounds
        #     bounds = np.asarray(
        #         (np.floor(bounds[0]), np.floor(bounds[1]), np.ceil(bounds[2]), np.ceil(bounds[3]))
        #     ).astype(np.uint16)
        #     data = self.X[bounds[1] : bounds[3], bounds[0] : bounds[2], ...]
        #     coords = np.stack(
        #         np.meshgrid(range(bounds[0], bounds[2] + 1), range(bounds[1], bounds[3] + 1)),
        #         axis=-1,
        #     ).reshape((-1, 2))
        #     mp = MultiPoint(coords) if coords.shape[0] > 1 else Point(coords)
        #     inters = np.asarray(mask.intersection(mp)).astype(np.uint16)
        #     if inters.size == 0:
        #         return inters
        #     else:
        #         data = data[inters[:, 1] - bounds[1], inters[:, 0] - bounds[0], ...]
        # else:
        #     data = self.X
        #
        # if genes is not None:
        #     if self.channel_names is None:
        #         data = data[..., genes]
        #     else:
        #         idx = np.argsort(self.channel_names)
        #         sorted_names = self.channel_names[idx]
        #         sorted_idx = np.searchsorted(sorted_names, genes)
        #         try:
        #             yidx = idx[sorted_idx]
        #         except IndexError:
        #             raise KeyError(
        #                 "elements {} not found".format(
        #                     [
        #                         genes[i]
        #                         for i in np.where(np.isin(genes, self.channel_names, invert=True))[
        #                             0
        #                         ]
        #                     ]
        #                 )
        #              )
        #         data = data[..., yidx]
        # return data

    @staticmethod
    def _encodingtype():
        return "fov-raster"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _set_backing(self, grp: Optional[h5py.Group]):
        super()._set_backing(grp)
        if grp is not None:
            assert isinstance(grp, h5py.Group)
            # self._backing should be reassigned from one of the caller functions (set_backing from BackableObject),
            # but to be safe let's set it to None explicity here
            self._backing = None
            self._write(grp)
            # self._X = None
        else:
            print('who is calling me?')
            assert self.isbacked

    def _write(self, grp):
        super()._write(grp)
        grp.create_dataset("X", data=self.X)
        # grp.create_dataset("X", data=self.X, compression="gzip", compression_opts=9)

    def _write_attributes_impl(self, obj):
        super()._write_attributes_impl(obj)
        if self._px_distance is not None:
            obj.attrs["px_distance"] = self._px_distance
        if self._px_dimensions is not None:
            obj.attrs["px_dimensions"] = self._px_dimensions

    def _plot_in_grid(
        self,
        channels_to_plot: list[str],
        grid_size: Union[int, list[int]] = 1,
        preprocessing: Optional[Callable] = None,
        cmap: Union[
            matplotlib.colors.Colormap, list[matplotlib.colors.Colormap]
        ] = matplotlib.cm.viridis,
        suptitle: Optional[str] = None,
    ):
        upper_limit_tiles = 50

        if (
            isinstance(grid_size, list)
            and len(grid_size) == 2
            and all(isinstance(x, int) for x in grid_size)
        ):
            n_tiles = grid_size[0] * grid_size[1]
        elif grid_size == 1 and len(channels_to_plot) != 1:
            n_tiles = len(channels_to_plot)
        elif isinstance(grid_size, int):
            n_tiles = grid_size ** 2
        else:
            raise ValueError(
                "'grid_size' must either be a single integer or a list of two integers."
            )
        if n_tiles > upper_limit_tiles:
            warnings.warn(
                "The generated plot will be very large. Consider plotting it outside of spatialmuon."
            )

        if len(channels_to_plot) > n_tiles:
            msg = "More channels available than covered by 'grid_size'. Only the first {} channels will be plotted".format(
                n_tiles
            )
            warnings.warn(msg)

        # Calcualte grid layout
        if not isinstance(grid_size, list):
            n_x = math.ceil(n_tiles ** 0.5)
            n_y = math.floor(n_tiles ** 0.5)
            if n_x * n_y < n_tiles:
                n_y += 1
        else:
            n_x = grid_size[0]
            n_y = grid_size[1]

        idx = get_channel_index_from_channel_name(self.var, channels_to_plot[0])
        (x, y) = self.X[:, :, idx].shape
        cell_size_x = 2 * x / max(x, y)
        cell_size_y = 2 * y / max(x, y)
        fig, axs = plt.subplots(n_y, n_x, figsize=(cell_size_y * n_y, cell_size_x * n_x))
        if len(channels_to_plot) > 1:
            axs = axs.flatten()

        for idx, channel in enumerate(channels_to_plot):
            if idx < n_tiles:
                # self._plot_in_canvas(
                #     channels_to_plot=[channel],
                #     preprocessing=preprocessing,
                #     cmap=cmap,
                #     ax=axs[idx],
                #     show_legend=False,
                #     show_colorbar=False,
                #     show_scalebar=idx == 0,
                # )
                self.plot(
                    channels=channel,
                    preprocessing=preprocessing,
                    cmap=cmap,
                    ax=axs[idx],
                    show_legend=False,
                    show_colorbar=False,
                    show_scalebar=idx == 0,
                )
        for idx in range(n_tiles, n_x * n_y):
            axs[idx].set_axis_off()
        if suptitle is not None:
            plt.suptitle(suptitle)
        plt.subplots_adjust()
        plt.tight_layout()
        plt.show()

    def _plot_in_canvas(
        self,
        channels_to_plot: list[str],
        preprocessing: Optional[Callable] = None,
        cmap: Union[
            matplotlib.colors.Colormap, list[matplotlib.colors.Colormap]
        ] = matplotlib.cm.viridis,
        ax: matplotlib.axes.Axes = None,
    ):
        for idx, channel in enumerate(channels_to_plot):
            a = 1 / (max(len(channels_to_plot) - 1, 2)) if idx > 0 else 1
            channel_index = get_channel_index_from_channel_name(self.var, channel)
            data_to_plot = self.X[:, :, channel_index]

            x = data_to_plot if preprocessing is None else preprocessing(data_to_plot)
            im = ax.imshow(x, cmap=cmap[idx], alpha=a)
        # im is used in the calling function in datatypes_utils.py to draw the show_colorbar, but we are not displaying a
        # show_colorbar when we have more than one channel, so let's return a nonsense value
        if len(channels_to_plot) == 1:
            return im
        else:
            return None

    def plot(
        self,
        channels: Optional[Union[str, list[str], int, list[int]]] = "all",
        grid_size: Union[int, list[int]] = 1,
        preprocessing: Optional[Callable] = None,
        overlap: bool = False,
        cmap: Union[
            matplotlib.colors.Colormap, list[matplotlib.colors.Colormap]
        ] = matplotlib.cm.viridis,
        ax: matplotlib.axes.Axes = None,
        show_title: bool = True,
        show_legend: bool = True,
        show_colorbar: bool = True,
        show_scalebar: bool = True,
        suptitle: Optional[str] = None,
    ):
        regions_raster_plot(
            self,
            channels=channels,
            grid_size=grid_size,
            preprocessing=preprocessing,
            overlap=overlap,
            cmap=cmap,
            ax=ax,
            show_title=show_title,
            show_legend=show_legend,
            show_colorbar=show_colorbar,
            show_scalebar=show_scalebar,
            suptitle=suptitle,
        )

    def __repr__(self):
        s = self.X.shape
        x, y = s[:2]
        c = s[-1]
        repr_str = f"{x}x{y} pixels image with {c} channels"
        return repr_str

    def accumulate_features(self, masks: "Masks"):
        return masks.accumulate_features(self)
