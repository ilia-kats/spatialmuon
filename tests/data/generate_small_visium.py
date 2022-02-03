##
# this file is not a test but just a quick script for us to generate "small_imc.h5smu"
import copy
import os
from pathlib import Path

import numpy as np

import spatialmuon as smu
from spatialmuon._core.anchor import Anchor

# this file is cumbersome because at the moment subsetting and copy constructors are not yet implemented

this_dir = Path(__file__).parent
fpath = this_dir / "small_visium.h5smu"
outfile = fpath
# outfile = 'tests/data/small_imc.h5smu'
# outfile = os.path.join(os.getcwd(), outfile)

f = "/data/spatialmuon/datasets/visium_mousebrain/smu/visium.h5smu"
d = smu.SpatialMuData(backing=f)
print(d)
a0 = d["Visium"]["ST8059049"]
a1 = d["Visium"]["ST8059049H&E"]

print(outfile)
if os.path.isfile(outfile):
    os.unlink(outfile)
new_smu = smu.SpatialMuData(outfile)
new_smu["visium"] = modality = smu.SpatialModality()
modality["expression"] = a0

d0 = modality["expression"]
# x_cutoff = 0
x_cutoff = 1400
y_cutoff = 1500
centers = d0.masks.masks_centers
obs_to_keep = (centers[:, 0] > x_cutoff) * (centers[:, 1] > y_cutoff)
print(f"keeping {np.sum(obs_to_keep)} out of {len(centers)} obs")
n_var_to_keep = 100
new_X = d0.X[obs_to_keep, :n_var_to_keep]
d0.var = d0.var.iloc[:n_var_to_keep]
d0.X = new_X
new_obs = d0.masks.obs[obs_to_keep]
d0.masks.obs = new_obs
d0.masks.masks_centers = d0.masks.masks_centers[np.where(obs_to_keep)]
d0.masks.masks_radii = d0.masks.masks_radii[np.where(obs_to_keep)]

vector = a1.anchor.vector
scale_factor = a1.anchor.scale_factor
img = a1.X[y_cutoff:, x_cutoff:, ...]
anchor = Anchor(origin=np.array([x_cutoff / scale_factor, y_cutoff / scale_factor]), vector=vector)
d1 = smu.Raster(X=img, anchor=anchor, coordinate_unit=a1.coordinate_unit)
new_smu.commit_changes_on_disk()
modality["image"] = d1

print("\nnew smu")
print(new_smu)
print(f"created at {outfile}")
print("filesize: ", end="")
os.system(f'ls -lh {outfile} | cut -d " " -f 5')
new_smu.backing.close()

print("\nnew smu repacked")
d = smu.SpatialMuData(backing=fpath)
d.repack(compression_level=9)
d = smu.SpatialMuData(backing=fpath)
print(d)
print("filesize: ", end="")
os.system(f'ls -lh {outfile} | cut -d " " -f 5')

print("")
print("")
