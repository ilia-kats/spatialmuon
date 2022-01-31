import unittest
import spatialmuon
from tests.testing_utils import initialize_testing
from spatialmuon._core.tiler import Tiles
import shutil
import os
import tempfile
import numpy as np
import copy
import matplotlib.pyplot as plt

PLOT = False
test_data_dir, DEBUGGING = initialize_testing()
fpath_imc = test_data_dir / "small_imc.h5smu"
fpath_visium = test_data_dir / "small_visium.h5smu"


def get_small_imc():
    with tempfile.TemporaryDirectory() as td:
        ##
        des = os.path.join(td, "small_imc.h5smu")
        shutil.copy(fpath_imc, des)
        d = spatialmuon.SpatialMuData(backing=des)
        if DEBUGGING and PLOT:
            _, ax = plt.subplots()
            d["imc"]["ome"].plot(0, ax=ax)
            d["imc"]["masks"].masks.plot(fill_colors=None, outline_colors="k", ax=ax)
            plt.show()
        return d


def get_small_imc_aligned():
    with tempfile.TemporaryDirectory() as td:
        ##
        des = os.path.join(td, "aligned_imc.h5smu")
        shutil.copy(fpath_imc, des)
        d = spatialmuon.SpatialMuData(backing=des)
        x = d["imc"]["ome"].X[...]
        assert x.shape == (40, 60, 10)
        bigger_x = np.zeros((100, 200, 10))
        bigger_x[60:, 140:, :] = x
        new_raster = spatialmuon.Raster(X=bigger_x)
        del d["imc"]["ome"]
        d["imc"]["ome"] = new_raster
        d["imc"]["masks"]._anchor = spatialmuon.Anchor(
            origin=np.array([140, 60]), vector=np.array([0.5, 0])
        )
        data = d["imc"]["masks"].masks.data
        # only even entries
        assert data.shape == (40, 60)
        data = data[np.ix_(*[range(0, i, 2) for i in data.shape])]
        assert data.shape == (20, 30)
        d["imc"]["masks"].masks._mask = data
        d['imc']['masks'].masks._obs = None
        d['imc']['masks'].masks._backing = None
        d['imc']['masks'].masks.update_obs_from_masks()
        new_regions = copy.copy(d["imc"]["masks"])
        del d["imc"]["masks"]
        d["imc"]["masks"] = new_regions
        ##
        if DEBUGGING and PLOT:
            _, ax = plt.subplots()
            d["imc"]["ome"].plot(0, ax=ax)
            d["imc"]["masks"].masks.plot(fill_colors=None, outline_colors="k", ax=ax)
            plt.show()
        return d


def get_small_visium():
    pass


if __name__ == "__main__":
    get_small_imc()
    get_small_imc_aligned()
    get_small_visium()
