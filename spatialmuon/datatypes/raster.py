from typing import Optional, Union, Literal
import warnings

import numpy as np
import h5py
from shapely.geometry import Polygon, Point, MultiPoint
from trimesh import Trimesh

from .. import FieldOfView
from ..utils import _get_hdf5_attribute


class Raster(FieldOfView):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        *,
        X: Optional[np.ndarray] = None,
        channel_names: Optional[list[str]] = None,
        px_dimensions: Optional[np.ndarray] = None,
        px_distance: Optional[np.ndarray] = None,
    ):
        if backing is not None:
            self._ndim = backing["X"].ndim - 1
            if self._ndim == 1:
                self._ndim = 2
            self._px_distance = _get_hdf5_attribute(backing.attrs, "px_distance")
            self._px_dimensions = _get_hdf5_attribute(backing.attrs, "px_dimensions")
            self._channel_names = _get_hdf5_attribute(backing.attrs, "channel_names")
            self._X = None
        else:
            if X is None:
                raise ValueError("no data and no backing store given")
            self._X = X
            self._ndim = X.ndim if X.ndim == 2 else X.ndim - 1
            if self._ndim < 2 or self._ndim > 3:
                raise ValueError("image dimensionality not supported")
            self._channel_names = channel_names
            self._px_dimensions = px_dimensions
            self._px_distance = px_distance
            if self._channel_names is not None:
                if (
                    self._X.ndim > 2
                    and len(self._channel_names) != self._X.shape[-1]
                    or self._X.ndim == 2
                    and len(self._channel_names) != 1
                ):
                    raise ValueError("channel names dimensionality is inconsistent with X")
            if self._px_dimensions is not None:
                self._px_dimensions = np.asarray(self._px_dimensions).squeeze()
                if self._px_dimensions.shape[0] != self._ndim or self._px_dimensions.ndim != 1:
                    raise ValueError("pixel_size dimensionality is inconsistent with X")
            if self._px_distance is not None:
                self._px_distance = np.asarray(self._px_distance).squeeze()
                if self._px_distance.shape[0] != self._ndim or self._px_distance.ndim != 1:
                    raise ValueError("pixel_distance dimensionality is inconsistent with X")
        super().__init__(backing)

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
    def channel_names(self) -> np.ndarray:
        return self._channel_names

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

    def subset(
        self,
        mask: Union[Polygon, Trimesh],
        polygon_method: Literal["project", "discard"] = "discard",
    ):
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
                        range(ub[0] - lb[0] + 2), range(ub[1] - lb[1] + 2), range(ub[2] - lb[2] + 2)
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
        mp = MultiPoint(
            np.stack(
                np.meshgrid(range(bounds[0], bounds[2] + 1), range(bounds[1], bounds[3] + 1)),
                axis=-1,
            ).reshape((-1, 2))
        )
        inters = np.asarray(mask.intersection(mp)).astype(np.uint16)
        if inters.size == 0:
            return inters
        else:
            return data[inters[:, 1] - bounds[1], inters[:, 0] - bounds[0], ...]

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
        grp.create_dataset("X", data=self.X, compression="gzip", compression_opts=9)

    def _write_attributes_impl(self, obj):
        super()._write_attributes_impl(obj)
        if self._px_distance is not None:
            obj.attrs["px_distance"] = self._px_distance
        if self._px_dimensions is not None:
            obj.attrs["px_dimensions"] = self._px_dimensions
        if self._channel_names is not None:
            obj.attrs["channel_names"] = self._channel_names
