import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import spatialmuon
from spatialmuon import SpatialModality, Raster
import tifffile
from xml.etree import ElementTree
import tempfile
from tests.testing_utils import initialize_testing

test_data_dir, DEBUGGING = initialize_testing()
fpath = test_data_dir / "ome_example.tiff"


class SpatialModality_creation(unittest.TestCase):
    def test_can_create_SpatialModality_from_Raster(self):
        ome = tifffile.TiffFile(fpath, is_ome=True)
        metadata = ElementTree.fromstring(ome.ome_metadata)[0]
        for chld in metadata:
            if chld.tag.endswith("Pixels"):
                metadata = chld
                break
        channel_names = []
        for channel in metadata:
            if channel.tag.endswith("Channel"):
                channel_names.append(channel.attrib["Fluor"])
        var = pd.DataFrame({"channel_name": channel_names})
        res = Raster(X=np.moveaxis(ome.asarray(), 0, -1), var=var, coordinate_unit="μm")

        mod = SpatialModality()
        mod["ome"] = res
        self.assertTrue(isinstance(mod, spatialmuon._core.spatialmodality.SpatialModality))

    def test_can_create_SpatialModality_from_Regions(self):
        tmp_dir_name = Path(tempfile.mkdtemp()) / "tmp.h5smu"
        # Create a small demo dataset
        np.random.seed(1000)
        N, D = 100, 20
        X = np.random.normal(size=(N, D))
        labels = [f"obs_{i}" for i in range(N)]
        obs = pd.DataFrame(index=labels)
        var = pd.DataFrame(index=[f"var_{i}" for i in range(D)])

        coords = np.abs(np.random.normal(size=(N, 2)))

        fovname = "myfov"

        radius = 1.0

        smudata = spatialmuon.SpatialMuData(tmp_dir_name)
        smudata["Visium"] = modality = spatialmuon.SpatialModality()

        masks = spatialmuon.ShapeMasks(
            masks_shape="circle", masks_centers=coords, masks_radii=radius, masks_labels=labels
        )
        cfov = spatialmuon.Regions(
            X=X,
            var=var,
            masks=masks,
            coordinate_unit="px",
        )
        modality[fovname] = cfov

        assert smudata["Visium"][fovname].n_var == D
        assert smudata["Visium"][fovname].masks.n_obs == N


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main()
    else:
        # SpatialModality_creation().test_can_create_SpatialModality_from_Raster()
        SpatialModality_creation().test_can_create_SpatialModality_from_Regions()
