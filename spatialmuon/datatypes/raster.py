from __future__ import annotations

from typing import Optional, Union, Literal, Callable, List, Dict, TYPE_CHECKING
import warnings

import math
import numpy as np
import h5py
import cv2
import copy
from shapely.geometry import Polygon, Point, MultiPoint
from trimesh import Trimesh
import matplotlib
import matplotlib.cm
import matplotlib.axes
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import pandas as pd
import math

from spatialmuon.utils import _get_hdf5_attribute
from spatialmuon.datatypes.datatypes_utils import (
    regions_raster_plot,
    get_channel_index_from_channel_name,
    PlottingMethod,
)
from spatialmuon._core.masks import Masks
from spatialmuon._core.anchor import Anchor
from spatialmuon._core.bounding_box import BoundingBox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar

from spatialmuon._core.fieldofview import FieldOfView
from spatialmuon.utils import _get_hdf5_attribute
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from spatialmuon.datatypes.regions import Regions


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
        kwargs["anchor"] = self.update_n_dim_in_anchor(
            ndim=len(X.shape) - 1 if X is not None else None, backing=backing, **kwargs
        )
        super().__init__(backing, **kwargs)
        if backing is not None:
            if X is not None or px_dimensions is not None or px_distance is not None:
                raise ValueError("attempting to specify properties for a non-empty backing store")
            self._X = self.backing["X"]
            self._px_dimensions = _get_hdf5_attribute(self.backing.attrs, "px_dimensions")
            self._px_distance = _get_hdf5_attribute(self.backing.attrs, "px_distance")
        else:
            if X is not None:
                self.X = X
            else:
                raise ValueError("no data and no backing store given")

            if px_dimensions is not None:
                self.px_dimensions = px_dimensions
                if len(self.px_dimensions.squeeze()) != self.ndim:
                    raise ValueError("px_dimensions dimensionality is inconsistent with X")
            else:
                self.px_dimensions = np.ones(self.ndim, np.uint8)

            if px_distance is not None:
                self.px_distance = px_distance
                if len(self.px_distance.squeeze()) != self.ndim:
                    raise ValueError("px_distance dimensionality is inconsistent with X")
            else:
                self.px_distance = np.ones(self.ndim, np.uint8)

            n_channels = self.X.shape[-1]
            if "var" not in kwargs:
                var = pd.DataFrame(dict(channel_name=range(n_channels)))
                self.var = var

    @property
    def X(self) -> Union[np.ndarray, h5py.Dataset]:
        return self._X

    @X.setter
    def X(self, new_X):
        self._X = new_X
        self.obj_has_changed("X")

    @property
    def px_dimensions(self) -> np.ndarray:
        return self._px_dimensions

    @px_dimensions.setter
    def px_dimensions(self, new_value):
        self._px_dimensions = new_value
        self.obj_has_changed("px_dimensions")

    @property
    def px_distance(self) -> np.ndarray:
        return self._px_distance

    @px_distance.setter
    def px_distance(self, new_value):
        self._px_distance = new_value
        self.obj_has_changed("px_distance")

    # TODO: this code is almost identical to the one in masks.py, do something to avoid redundancy
    @property
    def _untransformed_bounding_box(self) -> BoundingBox:
        assert self.ndim in [2, 3]
        if self.ndim == 3:
            raise NotImplementedError()
        h, w = self.X.shape[:2]
        px_dimensions = self.px_dimensions
        px_distance = self.px_distance
        actual_h = float(h) * px_dimensions[0] + (h - 1) * (px_distance[0] - 1)
        actual_w = float(w) * px_dimensions[1] + (w - 1) * (px_distance[1] - 1)
        bounding_box = BoundingBox(x0=0.0, y0=0.0, x1=actual_w, y1=actual_h)
        return bounding_box

    # # flake8: noqa: C901
    # def _getitem(
    #     self,
    #     mask: Optional[Union[Polygon, Trimesh]] = None,
    #     genes: Optional[Union[str, list[str]]] = None,
    #     polygon_method: Literal["project", "discard"] = "discard",
    # ):
    #     raise NotImplementedError()
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

    def _write_impl(self, grp):
        super()._write_impl(grp)
        if self.compressed_storage:
            if self.has_obj_changed("X"):
                if "X" in grp:
                    del grp["X"]
                grp.create_dataset("X", data=self.X, compression="gzip", compression_opts=9)
        else:
            if self.has_obj_changed("X"):
                if "X" in grp:
                    del grp["X"]
                grp.create_dataset("X", data=self.X)

    def _write_attributes_impl(self, obj):
        super()._write_attributes_impl(obj)
        if self.has_obj_changed("px_distance"):
            obj.attrs["px_distance"] = self.px_distance
        if self.has_obj_changed("px_dimensions"):
            obj.attrs["px_dimensions"] = self.px_dimensions

    # TODO: this function shares code with Regions._plot_in_grid(), unify better at some point putting for instance
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
        (x, y) = self.X[:, :, idx].shape
        cell_size_x = 2 * x / max(x, y)
        cell_size_y = 2 * y / max(x, y)
        fig, axs = plt.subplots(
            grid_size[1],
            grid_size[0],
            figsize=(cell_size_y * grid_size[1], cell_size_x * grid_size[0]),
        )
        if len(channels_to_plot) > 1:
            axs = axs.flatten()

        for idx, channel in enumerate(channels_to_plot):
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

    # TODO: code very similar to Regions._plot_in_canvas(), maybe unify
    # flake8: noqa: C901
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
    ):
        def get_crop(shape):
            if len(shape) != 3:
                raise ValueError("only the 2d case (H x W x C) is currently supported")
            nonlocal bounding_box
            if bounding_box is None:
                return (slice(None), slice(None))
            else:
                # bounding_box
                ubb = self._untransformed_bounding_box
                assert np.isclose(ubb.x0, 0.0)
                assert np.isclose(ubb.y0, 0.0)
                bb = self.anchor.inverse_transform_bounding_box(bounding_box)
                x_size = ubb.x1 - ubb.x0
                y_size = ubb.y1 - ubb.y0
                x0_relative = bb.x0 / x_size
                x1_relative = bb.x1 / x_size
                y0_relative = bb.y0 / y_size
                y1_relative = bb.y1 / y_size

                x0_real = round(shape[1] * x0_relative)
                x1_real = round(shape[1] * x1_relative)
                y0_real = round(shape[0] * y0_relative)
                y1_real = round(shape[0] * y1_relative)

                x0_real = max(0, x0_real)
                x1_real = min(shape[1], x1_real)
                y0_real = max(0, y0_real)
                y1_real = min(shape[0], y1_real)
                if x1_real <= 0 or y1_real <= 0 or x0_real >= shape[1] or y0_real >= shape[0]:
                    warnings.warn(
                        "the selected bounding box does not contain any portion of the raster image"
                    )
                    return None, None

                bb_real = BoundingBox(x0=x0_real, x1=x1_real, y0=y0_real, y1=y1_real)
                bb_real = self.anchor.transform_bounding_box(bb=bb_real)
                bounding_box = bb_real

                return slice(y0_real, y1_real), slice(x0_real, x1_real)

        def _imshow(x, alpha, cmap=None):
            if bounding_box is None:
                bb = self.bounding_box
            else:
                bb = bounding_box
            extent = np.array([bb.x0, bb.x1, bb.y0, bb.y1])
            im = ax.imshow(
                x, cmap=cmap, extent=extent, origin="lower", interpolation="none", alpha=alpha
            )
            return im

        if rgba:
            indices = [
                get_channel_index_from_channel_name(self.var, channel)
                for channel in channels_to_plot
            ]
            indices = np.array(indices)

            crop = get_crop(self.X.shape)
            if crop[0] is None or crop[1] is None:
                return None
            data_to_plot = self.X[crop[0], crop[1], indices]

            x = data_to_plot if preprocessing is None else preprocessing(data_to_plot)
            if np.min(x) < 0.0 or np.max(x) > 1.0:
                if x.dtype == np.uint8:
                    pass
                else:
                    warnings.warn(
                        "the data is not in the [0, 1] range. Plotting the result of an affine transformation to "
                        "make the data in [0, 1]. This is likely to cause different coloring if you are plotting a "
                        "subset of the data, as the colors are determined to the local values stracted to [0, 1]"
                    )
                    old_shape = x.shape
                    x = np.reshape(x, (-1, old_shape[-1]))
                    a = np.min(x, axis=0)
                    b = np.max(x, axis=0)
                    x = (x - a) / (b - a)
                    x = np.reshape(x, old_shape)
            im = _imshow(x=x, alpha=alpha)
        else:
            for idx, channel in enumerate(channels_to_plot):
                a = 1 / (max(len(channels_to_plot) - 1, 2)) if idx > 0 else 1
                channel_index = get_channel_index_from_channel_name(self.var, channel)
                crop = get_crop(self.X.shape)
                if crop[0] is None or crop[1] is None:
                    return None
                data_to_plot = self.X[crop[0], crop[1], channel_index]

                x = data_to_plot if preprocessing is None else preprocessing(data_to_plot)
                im = _imshow(x=x, alpha=a * alpha, cmap=cmap[idx])
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
        alpha: float = 1.0,
        bounding_box: Optional[BoundingBox] = None,
    ):
        regions_raster_plot(
            self,
            channels=channels,
            fill_color=None,
            outline_color=None,
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
        )

    def __repr__(self):
        s = self.X.shape
        y, x = s[:2]
        c = s[-1]
        repr_str = f"(Raster) {x}x{y} pixels image with {c} channels"
        return repr_str

    def accumulate_features(self, masks: Masks) -> Regions:
        from spatialmuon.processing.accumulator import accumulate_features

        return accumulate_features(masks=masks, fov=self)

    def extract_tiles(
        self,
        masks: Masks,
        tile_dim_in_units: Optional[float] = None,
        tile_dim_in_pixels: Optional[float] = None,
    ):
        from spatialmuon.processing.tiler import Tiles

        return Tiles(
            masks=masks,
            raster=self,
            tile_dim_in_units=tile_dim_in_units,
            tile_dim_in_pixels=tile_dim_in_pixels,
        )

    def scale_raster(
        self,
        factor: Optional[float] = None,
        target_w: Optional[int] = None,
        target_h: Optional[int] = None,
    ):
        b = np.array([factor is None, target_w is None, target_h is None])
        if not np.sum(b) == 2:
            raise ValueError(
                "one parameter is enough to determine the shape of the up/down-scaling"
            )
        h, w, _ = self.X.shape
        if target_w is not None:
            factor = target_w / w
        elif target_h is not None:
            factor = target_h / h

        x = self.X[...]
        s = np.array(x.shape)[:-1].astype(np.float)
        s *= factor
        s = np.floor(s).astype(int)
        channels = []
        for i in range(x.shape[-1]):
            c = cv2.resize(x[:, :, i], dsize=s.tolist()[::-1], interpolation=cv2.INTER_CUBIC)
            channels.append(c)
        new_X = np.stack(channels, axis=-1)

        self.X = new_X
        self.anchor.scale_vector(factor)
        self.commit_changes_on_disk()

    def crop(self, bounding_box: BoundingBox):
        ##
        real_bb = self.anchor.inverse_transform_bounding_box(bounding_box)
        x0 = int(max(real_bb.x0, 0))
        x1 = int(min(real_bb.x1, self.X.shape[1]))
        y0 = int(max(real_bb.y0, 0))
        y1 = int(min(real_bb.y1, self.X.shape[0]))

        new_X = self.X[y0:y1, x0:x1, :]
        untransformed_translation = np.array([x0, y0])
        transformed_translation = self.anchor.transform_coordinates(untransformed_translation)
        # to support for the legacy format in which .origin and .vector were of dtype in64
        self.anchor.origin = transformed_translation

        self.X = new_X
        self.commit_changes_on_disk()
