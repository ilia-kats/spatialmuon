from typing import Optional

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import h5py
from anndata._io.utils import read_attribute, write_attribute

from .. import FieldOfView, SpatialIndex


class SingleMolecule(FieldOfView):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        *,
        data: Optional[gpd.GeoDataFrame] = None,
        index_kwargs:dict={},
        **kwargs,
    ):
        if backing is not None:
            self._index = SpatialIndex(
                backing=backing["index"],
                dimension=backing["coordinates"].shape[1],
                **index_kwargs,
            )
        elif data is not None:
            self._data = data
            self._index = SpatialIndex(coordinates=np.vstack(data.geometry), **index_kwargs)
        else:
            raise ValueError("no coordinates and no backing store given")
        super().__init__(backing, **kwargs)

    @property
    def data(self):
        if self.isbacked:
            coords = read_attribute(self.backing["coordinates"])
            metadata = read_attribute(self.backing["metadata"])

            df = gpd.GeoDataFrame(metadata, geometry=[Point(*c) for c in coords])
            return df
        else:
            return self._data

    @property
    def ndim(self):
        if self.isbacked:
            return self.backing["coordinates"].shape[1]
        else:
            return self._data.shape[1]

    @staticmethod
    def _encoding() -> str:
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
                ranges.create_dataset(str(gene), data=[start, end])
