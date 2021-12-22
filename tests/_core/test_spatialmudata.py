import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import spatialmuon
from spatialmuon import Converter, SpatialMuData, SpatialModality
import tifffile
from xml.etree import ElementTree

# Get current file and pre-generate paths and names
this_dir = Path(__file__).parent
fpath_ome_example = this_dir / "../data/ome_example.tiff"
fpath_ome_mask_mouth = this_dir / "../data/mask_mouth.tiff"


class SpatialMuData_TestClass(unittest.TestCase):
    def test_can_create_SpatialMuData(self):
        this_dir = Path(__file__).parent
        fpath_h5smu_example = this_dir / "h5smu_example.h5smu"
        fpath_ome_example = this_dir / "../data/ome_example.tiff"
        fpath_ome_mask_left_eye = this_dir / "../data/mask_left_eye.tiff"
        fpath_ome_mask_right_eye = this_dir / "../data/mask_right_eye.tiff"
        fpath_ome_mask_mouth = this_dir / "../data/mask_mouth.tiff"

        c = Converter()

        smudata = SpatialMuData(fpath_h5smu_example, backingmode="r+")

        mod = SpatialModality(coordinate_unit="μm")
        mod["ome"] = c.raster_from_tiff(fpath_ome_example)
        mod["left_eye"] = c.rastermask_from_tiff(fpath_ome_mask_left_eye)
        mod["right_eye"] = c.rastermask_from_tiff(fpath_ome_mask_right_eye)
        mod["mouth"] = c.rastermask_from_tiff(fpath_ome_mask_mouth)

        smudata["IMC"] = mod
        self.assertTrue(isinstance(smudata, spatialmuon._core.spatialmudata.SpatialMuData))


if __name__ == "__main__":
    unittest.main()
