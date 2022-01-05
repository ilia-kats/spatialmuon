import unittest
import spatialmuon
from pathlib import Path

# Get current file and pre-generate paths and names
this_dir = Path(__file__).parent
fpath_ome = this_dir / "../data/ome_example.tiff"


class Raster_TestClass(unittest.TestCase):
    def test_is_created_with_anchor_dict(self):
        c = spatialmuon.Converter()
        ome_raster = c.raster_from_tiff(fpath_ome)
        self.assertTrue(isinstance(ome_raster.anchors, dict))

    def test_anchor_dict_contains_origin(self):
        c = spatialmuon.Converter()
        ome_raster = c.raster_from_tiff(fpath_ome)
        self.assertTrue("origin" in ome_raster.anchors.keys())


if __name__ == "__main__":
    unittest.main()
