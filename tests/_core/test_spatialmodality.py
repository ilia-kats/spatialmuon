import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from spatialmuon import SpatialModality
import tifffile
from xml.etree import ElementTree

# Get current file and pre-generate paths and names
this_dir = Path(__file__).parent
smily_fpath = this_dir /  "../data/smily.tiff"

class PlottingTestClass(unittest.TestCase):
    def test_SpatialModality_generation(self):
        ome = tifffile.TiffFile(smily_fpath)
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
        res = spatialmuon.Raster(
            X=np.moveaxis(ome.asarray(), 0, -1), 
            var=var
        )
        
        mod = SpatialModality(coordinate_unit="μm")
        mod["ome"] = res


if __name__ == "__main__":
    unittest.main()
