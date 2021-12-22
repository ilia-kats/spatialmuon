import unittest
import tifffile
import spatialmuon
from pathlib import Path

# Get current file and pre-generate paths and names
this_dir = Path(__file__).parent
fpath_ome = this_dir / "../data/ome_example.tiff"
fpath_ome_mask_left_eye = this_dir / "../data/mask_left_eye.tiff"
fpath_ome_mask_right_eye = this_dir / "../data/mask_right_eye.tiff"
fpath_ome_mask_mouth = this_dir / "../data/mask_mouth.tiff"


class Converter_TestClass(unittest.TestCase):
    def test_can_create_Raster_from_tiff(self):
        c = spatialmuon.Converter()
        ome_raster = c.raster_from_tiff(fpath_ome)
        self.assertTrue(isinstance(ome_raster, spatialmuon.datatypes.raster.Raster))

    def test_can_create_Raster_from_tiff(self):
        c = spatialmuon.Converter()
        ome_mask_left_eye = c.rastermask_from_tiff(fpath_ome_mask_left_eye)
        ome_mask_right_eye = c.rastermask_from_tiff(fpath_ome_mask_right_eye)
        ome_mask_mouth = c.rastermask_from_tiff(fpath_ome_mask_mouth)
        self.assertTrue(isinstance(ome_mask_left_eye, spatialmuon._core.masks.RasterMasks))
        self.assertTrue(isinstance(ome_mask_right_eye, spatialmuon._core.masks.RasterMasks))
        self.assertTrue(isinstance(ome_mask_mouth, spatialmuon._core.masks.RasterMasks))


if __name__ == "__main__":
    unittest.main()
