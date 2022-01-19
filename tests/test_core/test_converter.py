import os.path
import unittest
import spatialmuon
from pathlib import Path
import sys
from spatialmuon._core.converter import Converter
from tests.testing_utils import initialize_testing

test_data_dir, DEBUGGING = initialize_testing()
fpath_ome = test_data_dir / "ome_example.tiff"
fpath_ome_mask_left_eye = test_data_dir / "mask_left_eye.tiff"
fpath_ome_mask_right_eye = test_data_dir / "mask_right_eye.tiff"
fpath_ome_mask_mouth = test_data_dir / "mask_mouth.tiff"
fpath_small_imc = test_data_dir / "small_imc.h5smu"
print(test_data_dir)


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
