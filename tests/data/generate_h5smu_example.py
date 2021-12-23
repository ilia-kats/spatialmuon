from spatialmuon import Converter, SpatialMuData, SpatialModality
from pathlib import Path

this_dir = Path(__file__).parent
fpath_h5smu_example = this_dir / "h5smu_example.h5smu"
fpath_ome_example = this_dir / "ome_example.tiff"
fpath_ome_mask_left_eye = this_dir / "mask_left_eye.tiff"
fpath_ome_mask_right_eye = this_dir / "mask_right_eye.tiff"
fpath_ome_mask_mouth = this_dir / "mask_mouth.tiff"

c = Converter()

smudata = SpatialMuData(fpath_h5smu_example, backingmode="r+")

mod = SpatialModality(coordinate_unit="μm")
mod["ome"] = c.raster_from_tiff(fpath_ome_example)
mod["left_eye"] = c.rastermask_from_tiff(fpath_ome_mask_left_eye)
mod["right_eye"] = c.rastermask_from_tiff(fpath_ome_mask_right_eye)
mod["mouth"] = c.rastermask_from_tiff(fpath_ome_mask_mouth)

smudata["IMC"] = mod
