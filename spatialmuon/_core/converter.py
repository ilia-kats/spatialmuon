from __future__ import annotations

from typing import Optional, Union, List
import numpy as np
import pandas as pd
import tifffile
import spatialmuon
from xml.etree import ElementTree


class Converter:
    def raster_from_tiff(self, path) -> spatialmuon.datatypes.raster.Raster:
        """Opens a .tiff and converts it to a spm.Raster()."""

        ome = tifffile.TiffFile(path)
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
        res = spatialmuon.Raster(X=np.moveaxis(ome.asarray(), 0, -1), var=var)

        return res

    def rastermask_from_tiff(self, path) -> spatialmuon._core.masks.RasterMasks:
        """Opens a .tiff and converts it to a spm.RasterMask()."""

        masks = np.asarray(tifffile.imread(path))
        res = spatialmuon.RasterMasks(mask=masks)

        return res
