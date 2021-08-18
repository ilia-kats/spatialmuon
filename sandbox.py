#%%
import os
import sys
import spatialmuon as spm
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# matplotlib.use("cairo")

f = "/data/l989o/data/spatialmuon/imc/BaselTMA_SP41_15.475kx12.665ky_10000x8500_5_20170905_110_274_X13Y1_32_a0_full.h5smu"

a = spm.SpatialMuData(f, backingmode="r")
ax = plt.gca()
spm.plot.spatial(a["IMC"]["FOV"], ax)
#%%
plt.figure()
ax = plt.gca()
plt.show()

#%%
import os

os._exit(0)
#%%
#%%
#%%
ds = OmeDataset("train")
ome_index = 0
ds[ome_index].shape

fig, axes = plt.subplots(2, 2, figsize=(15, 7), constrained_layout=True)
axes = axes.flatten()
for i in range(4):
    ax = axes[i]
    im = ax.imshow(ds[0][:, :, i], cmap=matplotlib.cm.get_cmap("gray"))
    ax.set_xlabel("pixels")
    ax.set_ylabel("pixels")
    ax.set_title(f"channel {i} ({CHANNEL_NAMES[i]})")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.5)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_title("expression")
plt.suptitle(f"ome_index = {ome_index}")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 7), constrained_layout=True)
axes = axes.flatten()
for i in range(4):
    ax = axes[i]
    im = ax.imshow(np.arcsinh(ds[0][:, :, i]), cmap=matplotlib.cm.get_cmap("gray"))
    ax.set_xlabel("pixels")
    ax.set_ylabel("pixels")
    ax.set_title(f"channel {i} ({CHANNEL_NAMES[i]})")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.5)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_title("expression")
plt.suptitle(f"ome_index = {ome_index}")
plt.show()
