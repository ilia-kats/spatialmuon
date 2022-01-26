##
import scanpy as sc
import anndata as ad
import spatialmuon as smu
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import math

##
root = "/data/spatialmuon/datasets/tomoseq_ikmi"
os.listdir(os.path.join(root, "raw"))

##
# original files with all the counts, incl. background; same number of slices per animal;
f0 = os.path.join(root, "raw/tomoseq_positions_seurat_v3.h5ad")
# slices that were defined to belong to animals (not the background);
f1 = os.path.join(root, "raw/tomoseq_animals_seurat.h5ad")
# aligned data with the same amount of positions per animal — it was done using dynamic time warping based on regional markers to align all the samples to the reference one (b2-42-uncut, the highest quality sample we have).
f2 = os.path.join(root, "raw/tomoseq_all_seurat.h5ad")

img_root = os.path.join(root, "raw/polyp_images_jpg/")

if False:
    images = {}
    for file in os.listdir(img_root):
        if file.endswith(".jpg"):
            f = os.path.join(img_root, file)
            x = np.array(Image.open(f))
            images[file.replace(".jpg", "")] = x

    a0 = ad.read_h5ad(f0)
    a1 = ad.read_h5ad(f1)
    a2 = ad.read_h5ad(f2)

    ##
    print(a0)
    print(a1)
    print(a2)

    ##
    for sample_name, img in images.items():
        a0.obs["sample_name_str"] = a0.obs["sample_name"].apply(lambda x: x.decode("utf8"))
        to_keep0 = a0.obs["sample_name_str"] == sample_name
        n_slices0 = np.sum(to_keep0)
        aa0 = a0[to_keep0]

        # checking that we can use both sample_id and sample_name as keys
        n0 = len(a0.obs["sample_id"].unique())
        n1 = len(a0.obs["sample_name"].unique())
        n2 = len(a0.obs.apply(lambda x: f"{x.sample_id}_{x.sample_name}", axis=1).unique())
        assert n0 == n1
        assert n1 == n2
        d0 = dict(zip(a0.obs["sample_name"].tolist(), a0.obs["sample_id"].tolist()))
        d1 = dict(zip(a0.obs["sample_id"].tolist(), a0.obs["sample_name"].tolist()))

        a1.obs["sample_name_str"] = a1.obs["sample_name"].apply(lambda x: x.decode("utf8"))
        to_keep1 = a1.obs["sample_name_str"] == sample_name
        n_slices1 = np.sum(to_keep1)
        aa1 = a1[to_keep1]
        # confirmed, sample_id and sample_name are consistent
        debug = a1.obs["sample_name"].apply(lambda x: d0[x])
        assert np.prod((debug == a1.obs["sample_id"]).to_numpy()) == 1

        # let's add sample_name (the column is missing)
        a2.obs["sample_name"] = a2.obs["sample_id"].apply(lambda x: d1[x])
        a2.obs["sample_name_str"] = a2.obs["sample_name"].apply(lambda x: x.decode("utf8"))
        to_keep2 = a2.obs["sample_name_str"] == sample_name
        n_slices2 = np.sum(to_keep2)
        aa2 = a2[to_keep2]

        print(
            f"found {n_slices0} slices for sample '{sample_name}; {n_slices1}/{n_slices0} slices being under the animal'"
        )
        # if sample_name != 'b3-LWF-17-12hpa':
        #     continue
        if n_slices1 > 0:
            positions = aa0.obs.pos.tolist()
            assert np.allclose(positions, np.arange(1.0, n_slices0 + 1, 1))
            print(img.shape)
            masks = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint)
            begin = 0
            ends = np.linspace(0, img.shape[1], n_slices0 + 1)[1:]
            for i, float_end in enumerate(ends):
                end = math.floor(float_end)
                masks[:, begin:end] = i + 1
                begin = end
            outfile = os.path.join(root, "smu", f"{sample_name}.h5smu")
            s = smu.SpatialMuData(outfile, backingmode="w")
            mod = smu.SpatialModality()
            s["tomo-seq"] = mod
            raster = smu.Raster(X=img)
            raster_masks = smu.RasterMasks(mask=masks)
            expression_matrix = aa0.X.toarray()
            var = aa0.var.copy()
            var["channel_name"] = var.index.to_series().apply(lambda x: x.decode("utf8"))
            var.reset_index(inplace=True, drop=True)
            regions = smu.Regions(X=expression_matrix, masks=raster_masks, var=var)
            mod["expression"] = regions
            mod["image"] = raster

            ##
            # from skimage.segmentation import watershed
            # import cv2
            #
            # shifted = cv2.pyrMeanShiftFiltering(img, 0, 40)
            # shifted = shifted[:, :, 0]
            # threshold = 240
            # shifted[shifted > threshold] = 255
            # shifted[shifted <= threshold] = 0
            # plt.imshow(shifted)
            # plt.show()
            #
            # raster_masks_covering = smu.RasterMasks(mask=shifted)
            # regions_covering = smu.Regions(masks=raster_masks_covering)
            ##
            # determine which slices cover the animal
            # waiting for the new data, doing a proof of concepts
            # print(a1.obs.iloc[1])
            # print(a1.obs.iloc[2])
            # print(aa1.obs.slice_index.unique())
            # is_covering = np.array([False] * (n_slices0 + 1))
            # is_covering[9 + 1 : -11 + 1] = True
            # masks_not_covering = masks.copy()
            # indices = np.where(is_covering)[0]
            # for i in indices:
            #     cols = np.where(masks_not_covering == i)[1]
            #     masks_not_covering[:, cols] = 0
            # indices = np.where(np.logical_not(is_covering))[0]
            # for i in indices:
            #     cols = np.where(masks_not_covering == i)[1]
            #     slice_size = img.shape[1] / n_slices0
            #     border = round(img.shape[0] / 3)
            #     masks_not_covering[border:-border, cols] = 0
            # raster_masks_covering = smu.RasterMasks(mask=masks_not_covering)
            # regions_covering = smu.Regions(masks=raster_masks_covering)
            ##
            # fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
            # fig.subplots_adjust(hspace=0.20)
            # s["tomo-seq"]["image"].plot(ax=ax0)
            # s["tomo-seq"]["expression"].plot(11, ax=ax1, show_colorbar=False, show_scalebar=False)
            # regions_covering.masks.plot(fill_colors=None, outline_colors='w', ax=ax1)
            # plt.show()
            ##

            flip = False
            positions = aa2.obs.slice_index.tolist()
            if np.allclose(positions, np.arange(96, 0, -1)):
                flip = True
            else:
                assert np.allclose(positions, np.arange(1, 97, 1))
            img = np.array(Image.open(os.path.join(root, 'raw/ideal_nematostella.png')))
            print(img.shape)
            masks = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint)
            begin = 0
            ends = np.linspace(0, img.shape[0], 96 + 1)[1:]
            for i, float_end in enumerate(ends):
                end = math.floor(float_end)
                masks[begin:end, :] = i + 1
                begin = end

            masks[img[:, :, 0] == 255] = 0
            outfile = os.path.join(root, "smu", f"{sample_name}.h5smu")
            s = smu.SpatialMuData(outfile, backingmode="r+")
            mod = smu.SpatialModality()
            if 'time-warped' in s:
                del s['time-warped']
            s["time-warped"] = mod
            raster_masks = smu.RasterMasks(mask=masks)
            x = aa2.X.toarray()
            if flip:
                x = np.flipud(x)
            var = aa0.var.copy()
            var["channel_name"] = var.index.to_series().apply(lambda x: x.decode("utf8"))
            var = var[::-1]
            var.reset_index(inplace=True, drop=True)
            regions = smu.Regions(X=x, masks=raster_masks, var=var)
            mod["expression"] = regions
            ##
            # fig, ax = plt.subplots(1)
            # s["time-warped"]["expression"].plot(11, ax=ax, show_colorbar=True, show_scalebar=True)
            # ax.invert_yaxis()
            # ax.set_axis_off()
            # plt.show()
            # pass

##
# outfile = os.path.join(root, "smu", f"{sample_name}.h5smu")
# s = smu.SpatialMuData(outfile, backingmode="w")
l = os.listdir(os.path.join(root, 'smu'))
n = len(l)
fig, axes = plt.subplots(5, 4, figsize=(20, 20))
axes = axes.flatten()
axes[-1].set_axis_off()
gene_index = 6
for i, ll in enumerate(l):
    gene_name = s['time-warped']['expression'].var.iloc[gene_index]['gene_id']
    ax = axes[i]
    f = os.path.join(root, 'smu', ll)
    s = smu.SpatialMuData(f)
    s["time-warped"]["expression"].plot(gene_index, ax=ax, show_colorbar=True, show_scalebar=True)
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title(ll.replace('.h5smu', ''))
plt.suptitle(f'{gene_name} expression across samples (aligned with dynamic time-warping)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
pass