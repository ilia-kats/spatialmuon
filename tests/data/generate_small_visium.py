##
# this file is not a test but just a quick script for us to generate "small_imc.h5smu"
import spatialmuon as smu
import os
from pathlib import Path

this_dir = Path(__file__).parent
fpath = this_dir / "../data/small_visium.h5smu"
outfile = fpath
# outfile = 'tests/data/small_imc.h5smu'
# outfile = os.path.join(os.getcwd(), outfile)

f = "/data/spatialmuon/datasets/visium_mousebrain/smu/visium.h5smu"
d = smu.SpatialMuData(backing=f)
print(d)
ome = d["imc"]["ome"].X[...]
raster_masks = d["imc"]["masks"].masks._backing["imagemask"][...]
new_shape = (40, 60, 10)
new_ome = ome[: new_shape[0], : new_shape[1], : new_shape[2]]
new_raster_masks = raster_masks[: new_shape[0], : new_shape[1]]


print(outfile)
new_var = d["imc"]["ome"].var[: new_shape[2]]

if os.path.isfile(outfile):
    os.unlink(outfile)
new_smu = smu.SpatialMuData(outfile)
new_smu["imc"] = modality = smu.SpatialModality()
modality["ome"] = smu.Raster(X=new_ome, var=new_var, coordinate_unit="um")
raster_masks = smu.RasterMasks(mask=new_raster_masks)
raster_masks.update_obs_from_masks()
regions = smu.Regions(masks=raster_masks, coordinate_unit="um")
modality["masks"] = regions
print(new_smu)
