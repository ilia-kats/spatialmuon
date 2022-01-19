import unittest
import spatialmuon
from tests.testing_utils import initialize_testing
from spatialmuon._core.tiler import Tiles

test_data_dir, DEBUGGING = initialize_testing()
fpath = test_data_dir / "small_imc.h5smu"


class Tiler_TestClass(unittest.TestCase):
    def test_can_create_tiles_from_raster_masks(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        raster = d['imc']['ome']
        masks = d['imc']['masks'].masks
        t = Tiles(raster, masks, tile_dim=32)
        pass


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main()
    else:
        Tiler_TestClass().test_can_create_tiles_from_raster_masks()
