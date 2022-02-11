import copy
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

import spatialmuon
import spatialmuon as smu
from spatialmuon._core.anchor import Anchor
from tests.testing_utils import initialize_testing
import pandas as pd

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
        new_raster = spatialmuon.Raster(X=bigger_x, coordinate_unit=d["imc"]["ome"].coordinate_unit)
        del d["imc"]["ome"]
        d["imc"]["ome"] = new_raster
        del d["imc"]["masks"]["anchor"]
        d["imc"]["masks"].anchor = spatialmuon.Anchor(
            origin=np.array([140, 60]), vector=np.array([0.5, 0])
        )
        data = d["imc"]["masks"].masks.X[...]
        # only even entries
        assert data.shape == (40, 60)
        data = data[np.ix_(*[range(0, i, 2) for i in data.shape])]
        assert data.shape == (20, 30)
        d["imc"]["masks"].masks.X = data
        d["imc"]["masks"].masks.obs = pd.DataFrame()
        # print(d["imc"]["masks"].masks.obs)
        d["imc"]["masks"].masks.update_obs_from_masks()
        # print(d["imc"]["masks"].masks.obs)
        d.commit_changes_on_disk()
        # print(d["imc"]["masks"].masks.obs)
        ##
        d
        # new_regions = copy.copy(d["imc"]["masks"])
        # del d["imc"]["masks"]
        # d["imc"]["masks"] = new_regions
        ##
        if DEBUGGING and PLOT:
            _, ax = plt.subplots()
            d["imc"]["ome"].plot(0, ax=ax)
            d["imc"]["masks"].masks.plot(fill_colors=None, outline_colors="k", ax=ax)
            plt.show()
        ##
        return d


def get_small_visium():
    with tempfile.TemporaryDirectory() as td:
        ##
        des = os.path.join(td, "scaled_visium.h5smu")
        shutil.copy(fpath_visium, des)
        d = spatialmuon.SpatialMuData(backing=des)
        return d


def get_small_scaled_visium():
    with tempfile.TemporaryDirectory() as td:
        ##
        des = os.path.join(td, "scaled_visium.h5smu")
        shutil.copy(fpath_visium, des)
        s = spatialmuon.SpatialMuData(backing=des)

        image = s["visium"]["image"].X[...]
        im = Image.fromarray(image)

        im2x = im.resize((im.width * 2, im.height * 2), Image.ANTIALIAS)
        border = (30, 30, 30, 30)  # left, top, right, bottom
        im_crop = ImageOps.crop(im, border)
        im2x_crop = ImageOps.crop(im2x, border)

        origin = s["visium"]["image"].anchor.origin[...]
        scale_factor = s["visium"]["image"].anchor.scale_factor
        anchor2x = Anchor(origin=origin, vector=np.array([2.0, 0.0]) * scale_factor)
        anchor_crop = Anchor(
            origin=origin + np.array([30.0, 30.0]) / scale_factor,
            vector=np.array([1.0, 0.0]) * scale_factor,
        )
        anchor2x_crop = Anchor(
            origin=origin + np.array([15.0, 15.0]) / scale_factor,
            vector=np.array([2.0, 0.0]) * scale_factor,
        )

        mod = s["visium"]
        mod["image2x"] = smu.Raster(X=np.array(im2x), anchor=anchor2x)
        mod["image_crop"] = smu.Raster(X=np.array(im_crop), anchor=anchor_crop)
        mod["image2x_crop"] = smu.Raster(X=np.array(im2x_crop), anchor=anchor2x_crop)
        #
        # s_out['visium'] = new_mod
        # # plt.figure()
        # # plt.imshow(im)
        # # plt.imshow(im2x)
        # # plt.imshow(im_crop)
        # # plt.show()
        # print(s)
        return s


if __name__ == "__main__":
    # get_small_imc()
    # get_small_imc_aligned()
    # get_small_visium()
    get_small_scaled_visium()
