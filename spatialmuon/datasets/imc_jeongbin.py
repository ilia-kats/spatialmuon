##
import os
from PIL import Image
import numpy as np
import spatialmuon as smu
from spatialmuon._core.converter import Converter
import pandas as pd

folder = "/data/spatialmuon/datasets/imc_jeongbin/raw"
converter = Converter()

df = pd.read_csv(os.path.join(folder, "antibody_channel.csv"))

channels = []
for f in os.listdir(folder):
    if f.endswith(".tiff"):
        ff = os.path.join(folder, f)
        c = np.asarray(Image.open(ff))
        channels.append(c)
x = np.stack(channels)
x = np.moveaxis(x, 0, 2)

var = df.rename(columns={"Channel": "probe", "Label": "channel_name"})
var["channel_name"].tolist()
to_keep = np.logical_not(var["channel_name"].isna().to_numpy())
var = var[to_keep]
var.reset_index(inplace=True, drop=True)
x = x[:, :, to_keep]
raster = smu.Raster(X=x, var=var)
raster.plot(preprocessing=np.arcsinh)
raster.plot(preprocessing=np.arcsinh, channels="beta-Actin")

s = smu.SpatialMuData("/data/spatialmuon/datasets/imc_jeongbin/smu/imc.h5smu")
s["imc"] = smu.SpatialModality()
s["imc"]["ome"] = raster
