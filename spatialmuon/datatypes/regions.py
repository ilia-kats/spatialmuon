from __future__ import annotations

from typing import Optional, Union, Literal
from enum import Enum, auto
import warnings

import numpy as np
from scipy.sparse import spmatrix
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from trimesh import Trimesh
import h5py
from anndata import AnnData
from anndata._io.utils import read_attribute, write_attribute
from anndata._core.sparse_dataset import SparseDataset

from .. import FieldOfView, SpatialIndex
from ..utils import _read_hdf5_attribute, preprocess_3d_polygon_mask


class Regions(FieldOfView):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        *,
        X: Optional[Union[np.ndarray, spmatrix]] = None,
        obs: Optional[pd.DataFrame] = None,
        index_kwargs: dict = {},
        **kwargs,
    ):
        if backing is not None:
            self._index = SpatialIndex(
                backing=backing["index"], dimension=backing["coordinates"].shape[1], **index_kwargs
            )
            self._obs = read_attribute(backing["obs"])
            attrs = backing.attrs
        else:
            if X is None:
                raise ValueError("no expression data and no backing store given")
            else:
                self._X = X

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
    def obs(self) -> gpd.GeoDataFrame:
        return self._obs

    @property
    def n_obs(self) -> unt:
        return self._obs.shape[0]

    def _getitem(
        self,
        mask: Optional[Union[Polygon, Trimesh]] = None,
        genes: Optional[Union[str, list[str]]] = None,
        polygon_method: Literal["project", "discard"] = "discard",
    ) -> AnnData:
        if mask is not None:
            if self.ndim == 2:
                if not isinstance(mask, Polygon):
                    raise TypeError("Only polygon masks can be applied to 2D FOVs")
                idx = sorted(self._index.intersection(mask.bounds))
                obs = self.obs.iloc[idx, :].intersection(mask)
                X = self.X[idx, :][~obs.is_empty, :]
                obs = self.obs[~obs.is_empty]
                coords = np.vstack(obs.geometry)
                obs.drop(obs.geometry.name, axis=1, inplace=True)
            else:
                if isinstance(mask, Polygon):
                    bounds = preprocess_3d_polygon_mask(mask, self._coordinates, polygon_method)
                    idx = sorted(self._index.intersection(bounds))
                    sub = self._obs.iloc[idx, :].intersection(mask)
                    nemptyidx = ~sub.is_empty
                elif isinstance(mask, Trimesh):
                    idx = sorted(self._index.intersection(mask.bounds.reshape(-1)))
                    sub = self._obs.iloc[idx, :]
                    nemptyidx = mask.contains(np.vstack(sub.geometry))
                else:
                    raise TypeError("unknown mask type")
                X = (self.X[idx, :][nemptyidx, :],)
                obs = sub.iloc[nemptyidx, :]
        else:
            X = self.X[()]
            obs = self.obs
        ad = AnnData(X=X, obs=obs, var=self.var)
        if genes is not None:
            ad = ad[:, genes]
        return ad

    @property
    def ndim(self):
        if self.isbacked:
            return self.backing["coordinates"].shape[1]
        else:
            return self._coordinates.shape[1]

    @staticmethod
    def _encodingtype() -> str:
        return "fov-array"

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

    def _write_attributes_impl(self, obj):
        pass
