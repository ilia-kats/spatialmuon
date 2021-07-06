from typing import Optional, List

import geopandas as gpd
from shapely.geometry import Point
import h5py
from anndata._io.utils import read_attribute

from .. import FieldOfView

class SingleMolecule(FieldOfView):
    def __init__(self, *, data: Optional[gpd.GeoDataFrame]=None, **kwargs):
        super().__init__(**kwargs)
        self._data = data

    @property
    def data(self):
        coords = read_attribute(self._grp["coordinates"])
        genes = read_attribute(self._grp["feature_name"])
        metadata = read_attribute(self._grp["metadata"])

        df = gpd.GeoDataFrame(metadata, geometry=[Point(*c) for c in coords])
        df.set_axis(genes)

    @staticmethod
    def datatype() -> str:
        return "single-molecule"
