import unittest
import tifffile
from spatialmuon import Converter
from pathlib import Path

# Get current file and pre-generate paths and names
this_dir = Path(__file__).parent
fpath = this_dir / "../data/ome_example.tiff"


class Converter_TestClass(unittest.TestCase):
    def test_can_create_Raster_from_tiff(self):
        c = Converter()
        ome_raster = c.raster_from_tiff(fpath)
        self.assertTrue(isinstance(ome_raster, spatialmuon.datatypes.raster.Raster))


if __name__ == "__main__":
    unittest.main()
