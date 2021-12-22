##
import matplotlib.pyplot as plt

import spatialmuon as smu
import numpy as np

f = "/data/l989o/deployed/a/data/spatial_uzh_processed/a/spatialmuon/BaselTMA_SP41_15.475kx12.665ky_10000x8500_5_20170905_107_114_X13Y4_219_a0_full.h5smu"
d = smu.SpatialMuData(backing=f)
d["imc"]["ome"].plot(preprocessing=np.arcsinh)
##
d['imc']['masks'].plot()

##
import os
os._exit(0)
import matplotlib.pyplot as plt

x = d["imc"]["ome"].X[...]
x.shape[-1]
import matplotlib.gridspec as gridspec

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
