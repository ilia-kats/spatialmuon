##
import matplotlib.pyplot as plt
import pandas as pd

import spatialmuon as smu
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py

##
fpath = "/data/l989o/deployed/spatialmuon/tests/data/small_imc.h5smu"
f5 = h5py.File(fpath, 'r+')
print(f5['mod']['imc'].keys())
del f5['mod/imc/maximum']
del f5['mod/imc/mean']
f5.close()
##
f = "/data/l989o/deployed/a/data/spatial_uzh_processed/a/spatialmuon/BaselTMA_SP41_15.475kx12.665ky_10000x8500_5_20170905_107_114_X13Y4_219_a0_full.h5smu"
d = smu.SpatialMuData(backing=f)
ome = d['imc']['ome'].X[...]
ome.shape
raster_masks = d['imc']['masks'].masks._backing['imagemask'][...]
raster_masks.shape
new_shape = (40, 60, 10)
new_ome = ome[:new_shape[0], :new_shape[1], :new_shape[2]]
new_raster_masks = raster_masks[:new_shape[0], :new_shape[1]]
outfile = 'tests/data/small_imc.h5smu'
new_var = d['imc']['ome'].var[:new_shape[2]]
##
if os.path.isfile(outfile):
    os.unlink(outfile)
new_smu = smu.SpatialMuData(outfile)
new_smu["imc"] = modality = smu.SpatialModality(coordinate_unit="μm")
modality["ome"] = smu.Raster(X=new_ome, var=new_var)
raster_masks = smu.RasterMasks(mask=new_raster_masks)
raster_masks.update_obs_from_masks()
regions = smu.Regions(masks=raster_masks)
modality["masks"] = regions
print(new_smu)
##
x = d['imc']['masks'].masks._backing['imagemask'][...]
x.shape
new_shape = (40, 60)
new_x = x[:new_shape[0], :new_shape[1]]
new_x.shape
d['imc']['masks'].masks._backing['imagemasks'] = new_x
d['imc']['masks'].masks._obs = None
d['imc']['masks'].masks.update_obs_from_masks()
d['imc']['masks']
##
d["imc"]["ome"].plot(preprocessing=np.arcsinh)
##
##
d['imc']['masks'].plot()

##
os._exit(0)

x = d["imc"]["ome"].X[...]
x.shape[-1]

plt.figure(figsize=(30, 20))
gs = gridspec.GridSpec(11, 5)
gs.update(wspace=0., hspace=0.)  # set the spacing between axes.
for i in range(52):
    ax = plt.subplot(gs[i])
    ax.imshow(x[:, :, i])
for i in range(52, 55):
    ax = plt.subplot(gs[i])
    ax.set_axis_off()

plt.subplots_adjust()
plt.tight_layout()
plt.show()
