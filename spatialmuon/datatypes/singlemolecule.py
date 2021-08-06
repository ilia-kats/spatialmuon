import warnings
from typing import Optional, Union, Literal

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from trimesh import Trimesh
import h5py
from anndata._io.utils import read_attribute, write_attribute

from .. import FieldOfView, SpatialIndex
from ..utils import read_dataframe_subset


def _get_gdf(backing, yidx=None):
    if yidx is None:
        coords = read_attribute(backing["coordinates"])
        metadata = read_attribute(backing["metadata"])
    else:
        coords = backing["coordinates"][yidx, :]
        metadata = read_dataframe_subset(backing["metadata"], yidx)

    return gpd.GeoDataFrame(metadata, geometry=[Point(*c) for c in coords])


class SingleMolecule(FieldOfView):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        *,
        data: Optional[gpd.GeoDataFrame] = None,
        index_kwargs: dict = {},
        **kwargs,
    ):
        if backing is not None:
            self._index = SpatialIndex(
                backing=backing["index"],
                dimension=backing["coordinates"].shape[1],
                **index_kwargs,
            )
        elif data is not None:
            self._data = data.sort_index()
            self._index = SpatialIndex(coordinates=np.vstack(self._data.geometry), **index_kwargs)
        else:
            raise ValueError("no coordinates and no backing store given")
        super().__init__(backing, **kwargs)

    @property
    def data(self):
        return self._data_subset()

    def _data_subset(self, yidx=None):
        # only works with list[int] and 1D arrays at the moment
        if self.isbacked:
            if yidx is None:
                coords = read_attribute(self.backing["coordinates"])
                metadata = read_attribute(self.backing["metadata"])
            else:
                # reading a continuous slice is much faster than reading many individual elements
                yidx = np.asarray(yidx)
                min = yidx.min()
                slc = slice(min, yidx.max() + 1)
                yidx -= min
                coords = self.backing["coordinates"][slc][yidx]
                metadata = read_dataframe_subset(self.backing["metadata"], slc).iloc[yidx, :]

            return gpd.GeoDataFrame(metadata, geometry=[Point(*c) for c in coords])
        else:
            data = self._data
            if yidx is not None:
                data = data.iloc[yidx, :]
            return data

    def subset(
        self,
        mask: Union[Polygon, Trimesh],
        polygon_method: Literal["project", "discard"] = "discard",
    ):
        if self.ndim == 2:
            if not isinstance(mask, Polygon):
                raise TypeError("Only polygon masks can be applied to 2D FOVs")
            idx = sorted(self._index.intersection(mask.bounds))
            sub = self._data_subset(idx)
            inters = sub.intersection(mask)
            return sub[~inters.is_empty]
        else:
            if isinstance(mask, Polygon):
                if polygon_method == "discard":
                    bounds = list(mask.bounds)
                    bounds[2] = float("-Inf")
                    bounds[-1] = float("Inf")  # GeoPandas ignores the 3rd dimension
                elif polygon_method == "project":
                    if mask.has_z:
                        warnings.warn(
                            "method is `project` but mask has 3 dimensions. Assuming that mask is in-plane with the data and skipping projection."
                        )
                    else:
                        allcoords = np.vstack(self.data.coords)
                        mean = allcoords.mean(axis=0)
                        allcoords -= mean[:, np.newaxis]
                        cov = allcoords.T @ allcoords
                        projmat = np.linalg.eigh[cov][1][:, :2]
                        mask = Polygon(
                            np.asarray(mask.exterior.coords) @ projmat.T + mean[:, np.newaxis]
                        )
                        bounds = mask.bounds
                idx = sorted(self._index.intersection(bounds))
                sub = self._data_subset(idx).intersection(mask)
                return sub[~sub.is_empty]
            elif isinstance(mask, Trimesh):
                idx = sorted(self._index.intersection(mask.bounds.reshape(-1)))
                sub = self._data_subset(idx)
                return sub.iloc[mask.contains(np.vstack(sub.geometry)), :]

    @property
    def ndim(self):
        if self.isbacked:
            return self.backing["coordinates"].shape[1]
        else:
            return self._data.shape[1]

    @staticmethod
    def _encodingtype() -> str:
        return "fov-single-molecule"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _set_backing(self, value):
        super()._set_backing(value)
        if value is not None:
            self._write_data(value)
            self._index.set_backing(value, "index")
            self._data = None
        else:
            self._data = self.data
            self._index.set_backing(None)

    def _write(self, grp):
        super()._write(grp)
        self._write_data(grp)
        self._index.write(grp, "index")

    def _write_data(self, grp):
        if self._data is not None:
            grp.create_dataset(
                "coordinates",
                data=np.vstack(self._data.geometry),
                compression="gzip",
                compression_opts=9,
            )
            write_attribute(
                grp,
                "metadata",
                self._data.drop(self._data.geometry.name, axis=1),
                dataset_kwargs={"compression": "gzip", "compression_opts": 9},
            )

            ranges = grp.create_group("feature_range")
            genes, idx = np.unique(self._data.index.to_numpy(), return_counts=True)
            idx = np.hstack(([0], idx.cumsum()))
            for gene, start, end in zip(genes, idx[:-1], idx[1:]):
                ranges.create_dataset(str(gene), data=[start, end])
