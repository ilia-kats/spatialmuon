from typing import Optional, Union
from enum import Enum, auto
import warnings

import numpy as np
from scipy.sparse import spmatrix
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import h5py
from anndata._io.utils import read_attribute, write_attribute
from anndata._core.sparse_dataset import SparseDataset

from .. import FieldOfView, SpatialIndex
from spatialmuon.utils import _read_hdf5_attribute


class SpotShape(Enum):
    circle = auto()
    rectangle = auto()

    def __str__(self):
        return str(self.name)


class Array(FieldOfView):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        *,
        coordinates: Optional[np.ndarray] = None,
        X: Optional[Union[np.ndarray, spmatrix]] = None,
        var: Optional[pd.DataFrame] = None,
        obs: Optional[pd.DataFrame] = None,
        spot_shape: Optional[SpotShape] = None,
        spot_size: Optional[Union[float, np.ndarray]] = None,
        index_kwargs: dict = {},
        **kwargs,
    ):
        if backing is not None:
            self._index = SpatialIndex(
                backing=backing["index"], dimension=backing["coordinates"].shape[1], **index_kwargs
            )
            self._var = read_attribute(backing["var"])
            self._obs = read_attribute(backing["obs"])
            self._coordinates = read_attribute(backing["coordinates"])
            attrs = backing.attrs
            shape = _read_hdf5_attribute(attrs, "spot_shape")
            if shape == "circle":
                self._spot_shape = SpotShape.circle
            elif shape == "rectangle":
                self._spot_shape = SpotShape.rectangle
            else:
                raise ValueError(f"unknown spot shape {shape}")
            self._spot_size = _read_hdf5_attribute(attrs, "spot_size")
        else:
            if coordinates is None:
                raise ValueError("no coordinates and no backing store given")
            else:
                self._coordinates = coordinates
            if X is None:
                raise ValueError("no expression data and no backing store given")
            else:
                self._X = X
            if spot_shape is None:
                warnings.warn("spot shape not given, assuming circle")
                self._spot_shape = SpotShape.circle
            else:
                self._spot_shape = spot_shape
            if spot_size is None:
                raise ValueError("require spot size")
            else:
                self._spot_size = spot_size

            if var is not None:
                if var.shape[0] != self._X.shape[1]:
                    raise ValueError("X shape is inconsistent with var")
                else:
                    self._var = var
            else:
                self._var = pd.DataFrame(index=range(X.shape[1]))

            if obs is not None:
                if obs.shape[0] != self._X.shape[0]:
                    raise ValueError("X shape is inconsistent with obs")
                elif obs.shape[0] != self._coordinates.shape[0]:
                    raise ValueError("obs shape is inconsistent with coordinates")
                else:
                    self._obs = obs
            else:
                self._obs = pd.DataFrame(index=range(X.shape[0]))

            self._index = SpatialIndex(coordinates=self._coordinates, **index_kwargs)

        if self._spot_shape == SpotShape.circle and not np.isscalar(self._spot_size):
            raise ValueError("spot shape is circle, but spot size is not scalar")
        elif self._spot_shape == SpotShape.rectangle and (
            np.isscalar(self._spot_size)
            or self._spot_size.ndim > 1
            or self.spot_size.shape[0] != backing["coordinates"].shape[1]
        ):
            raise ValueError(
                "spot shape is rectangle, but spot size is not a 1D array of correct dimensions"
            )

        super().__init__(backing, **kwargs)

        self._obs = gpd.GeoDataFrame(self._obs, geometry=[Point(x) for x in self._coordinates])

    @property
    def X(self) -> Union[np.ndarray, spmatrix, h5py.Dataset, SparseDataset]:
        if self.isbacked:
            X = self.backing["X"]
            if isinstance(X, h5py.Group):
                return SparseDataset(X)
        else:
            return self._X

    @property
    def obs(self) -> pd.DataFrame:
        return self._obs

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @property
    def ndim(self):
        if self.isbacked:
            return self.backing["coordinates"].shape[1]
        else:
            return self._coordinates.shape[1]

    @staticmethod
    def _encoding() -> str:
        return "array"

    @staticmethod
    def _encodingversion() -> str:
        return "0.1.0"

    def _set_backing(self, obj):
        super()._set_backing(obj)
        if obj is not None:
            self._write_data(obj)
            self._index.set_backing(obj, "index")
            self._X = None
        else:
            self._X = read_attribute(obj, "X")
            self._index.set_backing(None)

    def _write(self, grp):
        super()._write(grp)
        self._write_data(grp)
        self._index.write(grp, "index")

    def _write_data(self, grp):
        write_attribute(
            grp, "X", self._X, dataset_kwargs={"compression": "gzip", "compression_opts": 9}
        )
        write_attribute(
            grp,
            "obs",
            self._obs.drop(self._obs.geometry.name, axis=1),
            dataset_kwargs={"compression": "gzip", "compression_opts": 9},
        )
        write_attribute(
            grp, "var", self._var, dataset_kwargs={"compression": "gzip", "compression_opts": 9}
        )
        write_attribute(
            grp,
            "coordinates",
            self._coordinates,
            dataset_kwargs={"compression": "gzip", "compression_opts": 9},
        )

    def _write_attributes_impl(self, obj):
        attrs = obj.attrs
        attrs["spot_shape"] = str(self._spot_shape)
        attrs["spot_size"] = self._spot_size
