import unittest
import spatialmuon
from pathlib import Path
from tests.testing_utils import initialize_testing
from tests.data.get_data import get_small_imc_aligned
import copy
import shutil
import os
import matplotlib.pyplot as plt

test_data_dir, DEBUGGING = initialize_testing()

fpath_ome = test_data_dir / "ome_example.tiff"


class Raster_TestClass(unittest.TestCase):
    def test_can_create_from_tiff(self):
        c = spatialmuon.Converter()
        ome_raster = c.raster_from_tiff(fpath_ome)
        self.assertTrue(isinstance(ome_raster, spatialmuon.Raster))

    def test_can_assign_to_SpatialModality(self):
        c = spatialmuon.Converter()
        ome_raster = c.raster_from_tiff(fpath_ome)
        mod = spatialmuon.SpatialModality()
        mod["ome"] = ome_raster
        self.assertTrue(isinstance(mod["ome"], spatialmuon.Raster))

    def test_can_crop_raster(self):
        d0 = get_small_imc_aligned()
        d1 = d0.clone()
        bb = spatialmuon.BoundingBox(x0=125, x1=200, y0=60, y1=110)
        plt.figure(figsize=(15, 4))

        ax = plt.subplot(1, 3, 1)
        d0["imc"]["ome"].plot(0, ax=ax)

        d1["imc"]["ome"].crop(bounding_box=bb)

        ax = plt.subplot(1, 3, 2)
        d0["imc"]["ome"].plot(0, ax=ax)

        ax = plt.subplot(1, 3, 3)
        d1["imc"]["ome"].plot(0, ax=ax, bounding_box=d0["imc"]["ome"].bounding_box)

        plt.show()
        ##
        print("ooo")

    def test_can_scale_raster(self):
        d0 = get_small_imc_aligned()
        d1 = d0.clone()
        d1["imc"]["ome"].scale_raster(factor=2.5)

        f0 = d0["imc"]["ome"]
        f1 = d1["imc"]["ome"]
        f2 = d0["imc"]["ome"].clone()
        f2.scale_raster(target_w=70)

        ##
        plt.figure(figsize=(15, 10))

        ax = plt.subplot(2, 3, 1)
        f0.plot(0, ax=ax)

        ax = plt.subplot(2, 3, 2)
        f1.plot(0, ax=ax)

        ax = plt.subplot(2, 3, 3)
        f2.plot(0, ax=ax)

        ax = plt.subplot(2, 3, 4)
        ax.imshow(f0.X[:, :, 0], origin="lower")

        ax = plt.subplot(2, 3, 5)
        ax.imshow(f1.X[:, :, 0], origin="lower")

        ax = plt.subplot(2, 3, 6)
        ax.imshow(f2.X[:, :, 0], origin="lower")

        plt.show()
        ##
        print("ooo")


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main(failfast=True)
    else:
        # Raster_TestClass().test_can_create_from_tiff()
        # Raster_TestClass().test_can_assign_to_SpatialModality()
        # Raster_TestClass().test_can_crop_raster()
        Raster_TestClass().test_can_scale_raster()
