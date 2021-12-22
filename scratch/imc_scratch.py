import spatialmuon as smu

#%%
f = "/data/l989o/deployed/a/data/spatial_uzh_processed/a/spatialmuon/BaselTMA_SP41_15.475kx12.665ky_10000x8500_5_20170905_107_114_X13Y4_219_a0_full.h5smu"
d = smu.SpatialMuData(backing=f)
d["imc"]["ome"].plot()


# %%
