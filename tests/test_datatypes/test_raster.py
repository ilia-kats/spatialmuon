import unittest
import spatialmuon
from pathlib import Path
from tests.testing_utils import initialize_testing

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


if __name__ == "__main__":
    if DEBUGGING:
        unittest.main()
    else:
        pass
