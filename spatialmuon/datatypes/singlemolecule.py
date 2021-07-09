from typing import Optional, List
import os

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import h5py
from anndata._io.utils import read_attribute, write_attribute

from .. import FieldOfView
from .. import SpatialIndex


class SingleMolecule(FieldOfView):
    def __init__(self, *, data: Optional[gpd.GeoDataFrame] = None, index_kwargs={}, **kwargs):
        super().__init__(**kwargs)
        self._data = data
        if self._data is not None:
            self._index = SpatialIndex(coordinates=np.vstack(data.geometry), **index_kwargs)
        elif self.isbacked:
            self._index = SpatialIndex(backing=self.backing["index"], dimension=self.backing["coordinates"].shape[1], **index_kwargs)
        else:
            self._index = None

    @property
    def data(self):
        if self.isbacked:
            coords = read_attribute(self.backing["coordinates"])
            metadata = read_attribute(self.backing["metadata"])

            df = gpd.GeoDataFrame(metadata, geometry=[Point(*c) for c in coords])
            return df
        else:
            return self._data

    @staticmethod
    def _encoding() -> str:
        return "single-molecule"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _write_attributes_impl(self, obj):
        super()._write_attributes_impl(obj)

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
        self._write_data(grp)
        if self._index is not None:
            self._index.write(grp, "index")

    def _write_data(self, grp):
        if self._data is not None:
            data = self._data.sort_index()
            grp.create_dataset(
                "coordinates", data=np.vstack(data.geometry), compression="gzip", compression_opts=9
            )
            write_attribute(
                grp,
                "metadata",
                self._data.drop(data.geometry.name, axis=1),
                dataset_kwargs={"compression": "gzip", "compression_opts": 9},
            )

            ranges = grp.create_group("feature_range")
            genes, idx = np.unique(data.index.to_numpy(), return_counts=True)
            idx = np.hstack(([0], idx.cumsum()))
            for gene, start, end in zip(genes, idx[:-1], idx[1:]):
                ranges.create_dataset(gene, data=[start, end])
