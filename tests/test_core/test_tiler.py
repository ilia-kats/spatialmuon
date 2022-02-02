import unittest

import spatialmuon
from spatialmuon._core.tiler import Tiles
from tests.data.get_data import get_small_imc, get_small_imc_aligned
from tests.testing_utils import initialize_testing

test_data_dir, DEBUGGING = initialize_testing()
fpath_imc = test_data_dir / "small_imc.h5smu"
fpath_visium = test_data_dir / "small_visium.h5smu"


class Tiler_TestClass(unittest.TestCase):
    def test_can_create_tiles_from_raster_masks(self):
        d = get_small_imc()
        raster = d["imc"]["ome"]
        masks = d["imc"]["masks"].masks
        t = Tiles(raster, masks, tile_dim=32)
        pass

    def test_can_create_tiles_from_aligned_raster_masks(self):
        d = get_small_imc_aligned()
        ##
        t = Tiles(raster=d["imc"]["ome"], masks=d["imc"]["masks"].masks, tile_dim=32)

    def test_can_create_tiles_from_shape_masks(self):
        self.skipTest("not implemented")
        return
        d = spatialmuon.SpatialMuData(backing=fpath_visium)
        raster = d["visium"]["image"]
        masks = d["visium"]["expression"].masks
        t = Tiles(raster, masks, tile_dim=32)
        pass


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main(failfast=True)
    else:
        Tiler_TestClass().test_can_create_tiles_from_raster_masks()
        # Tiler_TestClass().test_can_create_tiles_from_aligned_raster_masks()
        # Tiler_TestClass().test_can_create_tiles_from_shape_masks()
