from __future__ import annotations

from typing import Optional, Union, Literal, Callable
import warnings

import math
import numpy as np
import h5py
from shapely.geometry import Polygon, Point, MultiPoint
from trimesh import Trimesh
import matplotlib
import matplotlib.pyplot as plt

from .. import FieldOfView
from ..utils import _get_hdf5_attribute
from .._core.plot import plot_preview_grid


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

    def _getitem(
        self,
        mask: Optional[Union[Polygon, Trimesh]] = None,
        genes: Optional[Union[str, list[str]]] = None,
        polygon_method: Literal["project", "discard"] = "discard",
    ):
        if mask is not None:
            if self.ndim == 3:
                if isinstance(mask, Polygon):
                    if polygon_method == "project":
                        warnings.warn(
                            "Method `project` not possible with raster FOVs. Using `discard`."
                        )
                elif isinstance(mask, Trimesh):
                    lb, ub = np.floor(mask.bounds[0, :]), np.ceil(mask.bounds[1, :])
                    data = self.X[lb[0] : ub[0], lb[1] : ub[1], lb[2] : ub[2], ...]
                    coords = np.stack(
                        np.meshgrid(
                            range(ub[0] - lb[0] + 2),
                            range(ub[1] - lb[1] + 2),
                            range(ub[2] - lb[2] + 2),
                        ),
                        axis=1,
                    )
                    coords = coords[mask.contains(coords), :]
                    return data[coords[:, 1], coords[:, 0], coords[:, 2], ...]
            if not isinstance(mask, Polygon):
                raise TypeError("Only polygon masks can be applied to 2D FOVs")
            bounds = mask.bounds
            bounds = np.asarray(
                (np.floor(bounds[0]), np.floor(bounds[1]), np.ceil(bounds[2]), np.ceil(bounds[3]))
            ).astype(np.uint16)
            data = self.X[bounds[1] : bounds[3], bounds[0] : bounds[2], ...]
            coords = np.stack(
                np.meshgrid(range(bounds[0], bounds[2] + 1), range(bounds[1], bounds[3] + 1)),
                axis=-1,
            ).reshape((-1, 2))
            mp = MultiPoint(coords) if coords.shape[0] > 1 else Point(coords)
            inters = np.asarray(mask.intersection(mp)).astype(np.uint16)
            if inters.size == 0:
                return inters
            else:
                data = data[inters[:, 1] - bounds[1], inters[:, 0] - bounds[0], ...]
        else:
            data = self.X

        if genes is not None:
            if self.channel_names is None:
                data = data[..., genes]
            else:
                idx = np.argsort(self.channel_names)
                sorted_names = self.channel_names[idx]
                sorted_idx = np.searchsorted(sorted_names, genes)
                try:
                    yidx = idx[sorted_idx]
                except IndexError:
                    raise KeyError(
                        f"elements {[genes[i] for i in np.where(np.isin(genes, self.channel_names, invert=True))[0]]} not found"
                    )
                data = data[..., yidx]
        return data

    @staticmethod
    def _encodingtype():
        return "fov-raster"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _set_backing(self, obj):
        super()._set_backing(obj)
        if obj is not None:
            self._write(obj)
            self._X = None
        else:
            self._X = self.X

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
            
    def plot(
          self, 
          channels: Optional[Union[str, list[str]]] = "all",
          grid_size: Union[int, list[int]] = 1,
          preprocessing: Optional[Callable] = None,
          overlap: bool = False,
          cmap: Union[matplotlib.colors.Colormap, list[matplotlib.colors.Colormap]] = matplotlib.cm.viridis
        ):

        if not (isinstance(channels, list) or isinstance(channels, str)):
            raise ValueError("'channels' must be either a single character string or a list of them.")

        if isinstance(channels, list) and not all(isinstance(x, str) for x in channels):
            raise ValueError("If 'channels' is a list, all elements must be character strings.")

        if isinstance(channels, list):
            for c in channels:
                if c not in self.var["channel_name"].tolist():
                    raise ValueError("'{}' not found in channels, available are: {}".format(c, ', '.join(map(str, self.var["channel_name"]))))

        if isinstance(channels, str) and channels != "all":
            if channels not in self.var["channel_name"].tolist():
                raise ValueError("'{}' not found in channels, available are: {}".format(channels, ', '.join(map(str, self.var["channel_name"]))))

        if channels == "all":
            channels_to_plot = self.var["channel_name"].tolist()
        else:
            channels_to_plot = channels if isinstance(channels, list) else [channels]
            
        data_to_plot = {}    
            
        for idx, c in enumerate(channels_to_plot):
            channel_idx = self.var.query("channel_name == '{}'".format(c)).index.tolist()[0] # Get index of channel in data
            data_to_plot[c] = self.X[:,:,channel_idx]    
            
        plot_preview_grid(
            data_to_plot = data_to_plot,
            grid_size = grid_size,
            preprocessing = preprocessing,
            overlap = overlap,
            cmap = cmap
        )
