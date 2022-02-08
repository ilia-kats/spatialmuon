import unittest
import copy
import numpy as np

from spatialmuon.processing.tiler import Tiles
from tests.data.get_data import get_small_imc, get_small_imc_aligned, get_small_visium, get_small_scaled_visium
from tests.testing_utils import initialize_testing

test_data_dir, DEBUGGING = initialize_testing()
fpath_imc = test_data_dir / "small_imc.h5smu"
fpath_visium = test_data_dir / "small_visium.h5smu"


class Tiler_TestClass(unittest.TestCase):
    def test_can_create_masks_tiles_from_raster_masks(self):
        d = get_small_imc()
        masks = d["imc"]["masks"].masks
        masks.extract_tiles(tile_dim_in_units=None, tile_dim_in_pixels=32)
        pass

    def test_can_create_masks_tiles_from_aligned_raster_masks(self):
        d = get_small_imc_aligned()
        masks = d['imc']['masks'].masks
        masks.extract_tiles(tile_dim_in_units=20)

    def test_can_create_raster_tiles_from_raster_masks(self):
        d = get_small_imc()
        raster = copy.copy(d["imc"]["ome"])
        # to see what's going on
        new_X = d['imc']['masks'].masks.X[...][..., np.newaxis]
        raster.X = new_X
        masks = d["imc"]["masks"].masks
        raster.extract_tiles(masks=masks, tile_dim_in_pixels=32)

    def test_can_create_raster_tiles_from_aligned_raster_masks(self):
        d = get_small_imc_aligned()
        raster = copy.copy(d['imc']['ome'])
        new_X = raster.X[:, :, -1:]
        raster.X = new_X
        raster.extract_tiles(masks=d['imc']['masks'].masks, tile_dim_in_units=10)

    def test_can_create_raster_tiles_from_shape_masks(self):
        d = get_small_visium()
        raster = d["visium"]["image"]
        masks = d["visium"]["expression"].masks
        Tiles(masks=masks, raster=raster, tile_dim_in_units=55)


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main(failfast=True)
    else:
        Tiler_TestClass().test_can_create_masks_tiles_from_raster_masks()
        Tiler_TestClass().test_can_create_raster_tiles_from_raster_masks()
        Tiler_TestClass().test_can_create_masks_tiles_from_aligned_raster_masks()
        Tiler_TestClass().test_can_create_raster_tiles_from_aligned_raster_masks()
        Tiler_TestClass().test_can_create_raster_tiles_from_shape_masks()
