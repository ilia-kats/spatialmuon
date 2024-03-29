import unittest
import os
from pathlib import Path
import spatialmuon
from spatialmuon import Converter, SpatialMuData, SpatialModality, Regions
from tests.testing_utils import initialize_testing

test_data_dir, DEBUGGING = initialize_testing()
fpath_ome_example = test_data_dir / "ome_example.tiff"
fpath_ome_mask_mouth = test_data_dir / "mask_mouth.tiff"


class SpatialMuData_TestClass(unittest.TestCase):
    def test_can_create_SpatialMuData(self):
        this_dir = Path(__file__).parent
        fpath_h5smu_example = this_dir / "../data/h5smu_example.h5smu"
        fpath_ome_example = this_dir / "../data/ome_example.tiff"
        fpath_ome_mask_left_eye = this_dir / "../data/mask_left_eye.tiff"
        fpath_ome_mask_right_eye = this_dir / "../data/mask_right_eye.tiff"
        fpath_ome_mask_mouth = this_dir / "../data/mask_mouth.tiff"

        c = Converter()

        os.unlink(fpath_h5smu_example)
        smudata = SpatialMuData(fpath_h5smu_example, backingmode="w")

        mod = SpatialModality()
        mod["ome"] = c.raster_from_tiff(fpath_ome_example)
        mod["ome"].coordinate_units = "um"
        mod["left_eye"] = Regions(masks=c.rastermask_from_tiff(fpath_ome_mask_left_eye))
        mod["right_eye"] = Regions(masks=c.rastermask_from_tiff(fpath_ome_mask_right_eye))
        mod["mouth"] = Regions(masks=c.rastermask_from_tiff(fpath_ome_mask_mouth))

        smudata["IMC"] = mod
        self.assertTrue(isinstance(smudata, spatialmuon._core.spatialmudata.SpatialMuData))


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main(failfast=True)
    else:
        SpatialMuData_TestClass().test_can_create_SpatialMuData()
