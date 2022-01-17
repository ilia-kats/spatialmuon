##
# this file is not a test but just a quick script for us to generate "small_imc.h5smu"
import spatialmuon as smu
import os
import numpy as np
from pathlib import Path
import copy

# this file is cumbersome because at the moment subsetting and copy constructors are not yet implemented

this_dir = Path(__file__).parent
fpath = this_dir / "../data/small_visium.h5smu"
outfile = fpath
# outfile = 'tests/data/small_imc.h5smu'
# outfile = os.path.join(os.getcwd(), outfile)

f = "/data/spatialmuon/datasets/visium_mousebrain/smu/visium.h5smu"
d = smu.SpatialMuData(backing=f)
print(d)
a0 = d["Visium"]["ST8059049"]
a1 = d["Visium"]["ST8059049H&E"]

d0 = copy.copy(a0)
# x_cutoff = 0
x_cutoff = 1400
y_cutoff = 1500
centers = d0.masks._masks_centers
obs_to_keep = (centers[:, 0] > x_cutoff) * (centers[:, 1] > y_cutoff)
print(f"keeping {np.sum(obs_to_keep)} out of {len(centers)} obs")
new_X = d0.X[obs_to_keep, :]
d0._X = new_X
new_obs = d0.masks.obs[obs_to_keep]
d0.masks._obs = new_obs
d0.masks._masks_centers = d0.masks._masks_centers[np.where(obs_to_keep)]
d0.masks._masks_radii = d0.masks._masks_radii[np.where(obs_to_keep)]

img = a1.X[...]
d1 = smu.Raster(X=img)

print(outfile)
if os.path.isfile(outfile):
    os.unlink(outfile)
new_smu = smu.SpatialMuData(outfile)
new_smu["visium"] = modality = smu.SpatialModality()
modality["expression"] = d0
modality["image"] = d1
print(new_smu)
print(f"created at {outfile}")
