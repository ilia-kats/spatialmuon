import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import spatialmuon
from spatialmuon import SpatialModality, Raster
import tifffile
from xml.etree import ElementTree
import tempfile

# Get current file and pre-generate paths and names
this_dir = Path(__file__).parent
fpath = this_dir / "../data/ome_example.tiff"


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
        obs = pd.DataFrame(index=[f"obs_{i}" for i in range(N)])
        var = pd.DataFrame(index=[f"var_{i}" for i in range(D)])

        coords = np.abs(np.random.normal(size=(N, 2)))

        fovname = "myfov"
        fovidx = 0

        radius = 1.0

        smudata = spatialmuon.SpatialMuData(tmp_dir_name)
        smudata["Visium"] = modality = spatialmuon.SpatialModality()

        spots_dict = {o: ((x, y), radius) for (o, (x, y)) in zip(obs.index.tolist(), coords)}
        masks = spatialmuon.ShapeMasks(masks_dict=spots_dict, obs=obs)
        cfov = spatialmuon.Regions(
            X=X,
            var=var,
            translation=[0, 0, fovidx * 10],
            scale=1.23,
            masks=masks,
            coordinate_unit="px",
        )
        modality[fovname] = cfov

        assert smudata["Visium"][fovname].n_var == D
        assert smudata["Visium"][fovname].masks.n_obs == N


if __name__ == "__main__":
    unittest.main()
