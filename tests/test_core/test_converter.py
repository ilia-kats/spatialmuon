import os.path
import unittest
import spatialmuon
from pathlib import Path
import sys
from spatialmuon._core.converter import Converter

DEBUGGING = False
try:
    __file__
except NameError as e:
    if str(e) == "name '__file__' is not defined":
        DEBUGGING = True
    else:
        raise e
if sys.gettrace() is not None:
    DEBUGGING = True

if not DEBUGGING:
    # Get current file and pre-generate paths and names
    this_dir = Path(__file__).parent
else:
    this_dir = Path(os.path.expanduser("~/spatialmuon/tests/data/"))

# Get current file and pre-generate paths and names
fpath_ome = this_dir / "../data/ome_example.tiff"
fpath_ome_mask_left_eye = this_dir / "../data/mask_left_eye.tiff"
fpath_ome_mask_right_eye = this_dir / "../data/mask_right_eye.tiff"
fpath_ome_mask_mouth = this_dir / "../data/mask_mouth.tiff"
fpath_small_imc = this_dir / "../data/small_imc.h5smu"


class Converter_TestClass(unittest.TestCase):
    def test_can_create_Raster_from_tiff(self):
        c = spatialmuon.Converter()
        ome_raster = c.raster_from_tiff(fpath_ome)
        self.assertTrue(isinstance(ome_raster, spatialmuon.datatypes.raster.Raster))

    def test_can_create_RasterMask_from_tiff(self):
        c = spatialmuon.Converter()
        ome_mask_left_eye = c.rastermask_from_tiff(fpath_ome_mask_left_eye)
        ome_mask_right_eye = c.rastermask_from_tiff(fpath_ome_mask_right_eye)
        ome_mask_mouth = c.rastermask_from_tiff(fpath_ome_mask_mouth)
        self.assertTrue(isinstance(ome_mask_left_eye, spatialmuon._core.masks.RasterMasks))
        self.assertTrue(isinstance(ome_mask_right_eye, spatialmuon._core.masks.RasterMasks))
        self.assertTrue(isinstance(ome_mask_mouth, spatialmuon._core.masks.RasterMasks))

    def test_can_create_AnnData_from_Regions(self):
        d = spatialmuon.SpatialMuData(backing=fpath_small_imc)
        accumulated = d["imc"]["ome"].accumulate_features(d["imc"]["masks"].masks)
        e = accumulated["mean"]
        adata = Converter().regions_to_anndata(e)
        import scanpy

        scanpy.pl.spatial(adata, spot_size=3)


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main()
    else:
        Converter_TestClass().test_can_create_Raster_from_tiff()
        Converter_TestClass().test_can_create_RasterMask_from_tiff()
        Converter_TestClass().test_can_create_AnnData_from_Regions()
