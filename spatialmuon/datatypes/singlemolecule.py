from __future__ import annotations

import warnings
from typing import Optional, Union, Literal

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from ncls import NCLS
from trimesh import Trimesh
import h5py
from anndata._io.utils import read_attribute, write_attribute

from .. import FieldOfView, SpatialIndex
from ..utils import read_dataframe_subset, preprocess_3d_polygon_mask


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

    def _data_subset(self, yidx=None, genes=None):
        # only works with list[int] and 1D arrays at the moment
        if self.isbacked:
            if yidx is None:
                if genes is not None:
                    if isinstance(genes, str):
                        genes = [genes]
                    coords = []
                    metadata = []
                    for g in genes:
                        rng = self.backing["feature_range"][g][()]
                        coords.append(self.backing["coordinates"][rng[0] : rng[1]])
                        metadata.append(
                            read_dataframe_subset(self.backing["metadata"], slice(rng[0], rng[1]))
                        )
                    coords = np.vstack(coords)
                    metadata = pd.concat(metadata, axis=0)
                else:
                    coords = read_attribute(self.backing["coordinates"])
                    metadata = read_attribute(self.backing["metadata"])
            else:
                yidx = np.asarray(yidx)
                if genes is not None:
                    if isinstance(genes, list) and len(genes) == 1:
                        genes = genes[0]
                    if isinstance(genes, str):
                        rng = self.backing["feature_range"][genes][()]
                        ncls = NCLS(rng[0, np.newaxis], rng[1, np.newaxis] - 1, rng[0, np.newaxis])
                    else:
                        intervals = []
                        for g in genes:
                            rng = self.backing["feature_range"][g][()]
                            intervals.append(rng)
                        intervals = np.vstack(intervals)
                        ncls = NCLS(intervals[:, 0], intervals[:, 1] - 1, intervals[:, 0])
                    idx = yidx if yidx.size > 1 else yidx[np.newaxis]
                    yidx = yidx[ncls.has_overlaps(idx, idx, np.arange(yidx.size))]

                # reading a continuous slice is much faster than reading many individual elements
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
            if genes is not None:
                data = data.loc[genes, :]
            return data

    def _getitem(
        self,
        mask: Optional[Union[Polygon, Trimesh]] = None,
        genes: Optional[Union[str, list[str]]] = None,
        polygon_method: Literal["discard", "project"] = "discard",
    ):
        if mask is not None:
            if self.ndim == 2:
                if not isinstance(mask, Polygon):
                    raise TypeError("Only polygon masks can be applied to 2D FOVs")
                idx = sorted(self._index.intersection(mask.bounds))
                sub = self._data_subset(idx, genes)
                inters = sub.intersection(mask)
                return sub[~inters.is_empty]
            else:
                if isinstance(mask, Polygon):
                    bounds = preprocess_3d_polygon_mask(
                        mask, np.vstack(self.data.geometry), polygon_method
                    )
                    idx = sorted(self._index.intersection(bounds))
                    sub = self._data_subset(idx, genes).intersection(mask)
                    return sub[~sub.is_empty]
                elif isinstance(mask, Trimesh):
                    idx = sorted(self._index.intersection(mask.bounds.reshape(-1)))
                    sub = self._data_subset(idx, genes)
                    return sub.iloc[mask.contains(np.vstack(sub.geometry)), :]
                else:
                    raise TypeError("unknown masks type")
        elif genes is not None:
            return self._data_subset(genes=genes)

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
