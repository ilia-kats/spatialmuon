import unittest
import spatialmuon
from pathlib import Path

# Get current file and pre-generate paths and names
this_dir = Path(__file__).parent
fpath_ome = this_dir / "../data/ome_example.tiff"


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
    unittest.main()
