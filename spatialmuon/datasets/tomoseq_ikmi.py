##
import pandas as pd
import scanpy as sc
import anndata as ad
import spatialmuon as smu
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
from PIL import Image
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid

##
root = "/data/spatialmuon/datasets/tomoseq_ikmi"
os.listdir(os.path.join(root, "raw"))

##
# original files with all the counts, incl. background; same number of slices per animal;
f0 = os.path.join(root, "raw/tomoseq_all_seurat.h5ad")
# slices that were defined to belong to animals (not the background);
f1 = os.path.join(root, "raw/tomoseq_animals_seurat.h5ad")
# aligned data with the same amount of positions per animal — it was done using dynamic time warping based on regional markers to align all the samples to the reference one (b2-42-uncut, the highest quality sample we have).
f2 = os.path.join(root, "raw/tomoseq_positions_seurat_v3.h5ad")

img_root = os.path.join(root, "raw/polyp_images_jpg/")

manual_alignments = {
                'b3-LWF-18-24hpa': [116, 425],
                'b3-LWF-13-96hpa': [78, 432],
                'b3-LWF-17-12hpa': [50, 398]
            }

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
        a2.obs["sample_name_str"] = a2.obs["sample_name"].apply(lambda x: x.decode("utf8"))
        to_keep2 = a2.obs["sample_name_str"] == sample_name
        n_slices2 = np.sum(to_keep2)
        aa2 = a2[to_keep2]

        # checking that we can use both sample_id and sample_name as keys
        n2 = len(a2.obs["sample_id"].unique())
        n1 = len(a2.obs["sample_name"].unique())
        n2 = len(a2.obs.apply(lambda x: f"{x.sample_id}_{x.sample_name}", axis=1).unique())
        assert n2 == n1
        assert n1 == n2
        d2 = dict(zip(a2.obs["sample_name"].tolist(), a2.obs["sample_id"].tolist()))
        d1 = dict(zip(a2.obs["sample_id"].tolist(), a2.obs["sample_name"].tolist()))

        a1.obs["sample_name_str"] = a1.obs["sample_name"].apply(lambda x: x.decode("utf8"))
        to_keep1 = a1.obs["sample_name_str"] == sample_name
        n_slices1 = np.sum(to_keep1)
        aa1 = a1[to_keep1]
        # confirmed, sample_id and sample_name are consistent
        debug = a1.obs["sample_name"].apply(lambda x: d2[x])
        assert np.prod((debug == a1.obs["sample_id"]).to_numpy()) == 1

        # let's add sample_name (the column is missing)
        a0.obs["sample_name"] = a0.obs["sample_id"].apply(lambda x: d1[x])
        a0.obs["sample_name_str"] = a0.obs["sample_name"].apply(lambda x: x.decode("utf8"))
        to_keep0 = a0.obs["sample_name_str"] == sample_name
        n_slices0 = np.sum(to_keep0)
        aa0 = a0[to_keep0]

        print(
            f"found {n_slices0} slices for sample '{sample_name}; {n_slices1}/{n_slices0} slices being under the animal'"
        )
        # if sample_name != 'b3-LWF-17-12hpa':
        #     continue
        if n_slices1 > 0:
            ##
            # process the first data frame (all the slices, unaligned to the figures)
            flip = False
            positions = aa0.obs.slice_index.tolist()
            if np.allclose(positions, np.arange(96, 0, -1)):
                flip = True
            else:
                assert np.allclose(positions, np.arange(1, 97, 1))
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
            img = img / 255
            raster = smu.Raster(X=img)
            raster_masks = smu.RasterMasks(mask=masks)
            expression_matrix = aa0.X.toarray()
            var = aa0.var.copy()
            var["channel_name"] = var.index.to_series().apply(lambda x: x.decode("utf8"))
            if flip:
                var = var[::-1]
                expression_matrix = np.flipud(expression_matrix)
            var.reset_index(inplace=True, drop=True)
            regions = smu.Regions(X=expression_matrix, masks=raster_masks, var=var)
            ##
            # processed the second data frame (only slices under the animal)
            u = aa1.obs.slice_index.unique()
            min_index = u.min()
            max_index = u.max()
            obs = regions.masks.obs
            obs['covering_the_animal'] = False
            obs.loc[obs.original_labels.isin(u), 'covering_the_animal'] = True
            accumulated = raster.accumulate_features(raster_masks)
            slide_width = img.shape[1] / n_slices0
            min_boundary = obs[obs.original_labels == min_index].region_center_y.item() - slide_width / 2
            max_boundary = obs[obs.original_labels == max_index].region_center_y.item() + slide_width / 2

            pixel_boundaries = dict(zip(images.keys(), [[0, 600]] * len(images)))
            for k, v in manual_alignments.items():
                pixel_boundaries[k] = v

            new_anchor = regions.anchor.map_untransformed_to_untransformed_fov(raster, source_points=np.array([
                [min_boundary, 0],
                [max_boundary, 0]
            ]), target_points=np.array([
                [pixel_boundaries[sample_name][0], 0],
                [pixel_boundaries[sample_name][1], 0]
            ]))
            regions._anchor = new_anchor
            mod["expression"] = regions
            mod["image"] = raster
            pass
            ##
            # fig, (ax0, ax1) = plt.subplots(2, 1, constrained_layout=True, figsize=(5, 10))
            if False or sample_name == 'b3-LWF-17-12hpa':
                fig = plt.figure()
                ax0, ax1 = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0.1)
                ax0.sharex(ax1)
                # fig.subplots_adjust(hspace=0.20)
                s["tomo-seq"]["image"].plot(ax=ax0)
                ax0.axvline(x=pixel_boundaries[sample_name][0], c='r', linestyle='--')
                ax0.axvline(x=pixel_boundaries[sample_name][1], c='r', linestyle='--')
                ax0.set_title(sample_name)
                s["tomo-seq"]["expression"].plot(11, ax=ax1, show_colorbar=False, show_scalebar=False, show_title=False)
                # regions_covering.masks.plot(fill_colors=None, outline_colors='w', ax=ax1)
                plt.suptitle(f'flip = {flip}')
                # l_x = (200, 450)
                # l_y = (30, 130)
                # ax0.set_ylim(l_y)
                # ax1.set_ylim(l_y)
                # ax0.set_xlim(l_x)
                # ax1.set_xlim(l_x)
                plt.show()
            ##
            if False:
                fig, ax = plt.subplots(1)
                s["tomo-seq"]["image"].plot(ax=ax)
                ax.axvline(x=pixel_boundaries[sample_name][0], c='r', linestyle='--')
                ax.axvline(x=pixel_boundaries[sample_name][1], c='r', linestyle='--')
                s["tomo-seq"]["expression"].plot(11, ax=ax, show_colorbar=False, show_scalebar=False, alpha=0.5)
                # regions_covering.masks.plot(fill_colors=None, outline_colors='w', ax=ax1)
                # plt.xlim((150, 600))
                plt.show()
                print('')
            pass


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
            positions = aa2.obs.pos.tolist()
            assert np.allclose(positions, np.arange(1, n_slices2 + 1, 1))
            f = os.path.join(root, 'raw/ideal_nematostella.png')
            ideal_nematostella = np.array(Image.open(f))
            masks = np.zeros((ideal_nematostella.shape[0], ideal_nematostella.shape[1]), dtype=np.uint)
            begin = 0
            ends = np.linspace(0, ideal_nematostella.shape[0], n_slices2 + 1)[1:]
            for i, float_end in enumerate(ends):
                end = math.floor(float_end)
                masks[begin:end, :] = i + 1
                begin = end
            ##
            masks[ideal_nematostella[:, :, 0] == 255] = 0

            expression_matrix = aa2.X.toarray()
            var = aa2.var.copy()
            var["channel_name"] = var.index.to_series().apply(lambda x: x.decode("utf8"))
            var.reset_index(inplace=True, drop=True)

            outfile = os.path.join(root, "smu", f"{sample_name}.h5smu")
            s = smu.SpatialMuData(outfile)
            mod = smu.SpatialModality()
            if 'time-warped' in s:
                del s['time-warped']
            s["time-warped"] = mod
            raster_masks = smu.RasterMasks(mask=masks)
            regions = smu.Regions(X=expression_matrix, masks=raster_masks, var=var)

            mod["expression"] = regions

            # fig, ax = plt.subplots(1)
            # s["time-warped"]["expression"].plot(11, ax=ax, show_colorbar=True, show_scalebar=True)
            # ax.invert_yaxis()
            # ax.set_axis_off()
            # plt.show()

            pass


##
# outfile = os.path.join(root, "smu", f"{sample_name}.h5smu")
# s = smu.SpatialMuData(outfile, backingmode="w")
if False:
    l = os.listdir(os.path.join(root, 'smu'))
    n = len(l)
    fig, axes = plt.subplots(5, 4, figsize=(20, 20))
    axes = axes.flatten()
    im = Image.open(os.path.join(root, 'raw/ideal_nematostella_comparison.png'))
    axes[0].imshow(im)
    axes[0].axis("off")
    gene_index = 6
    for i, ll in enumerate(l):
        f = os.path.join(root, 'smu', ll)
        s = smu.SpatialMuData(f)
        gene_name = s['time-warped']['expression'].var.iloc[gene_index]['gene_id']
        ax = axes[i + 1]
        s["time-warped"]["expression"].plot(gene_index, ax=ax, show_colorbar=False, show_scalebar=True)
        ax.invert_yaxis()
        ax.set_axis_off()
        ax.set_title(ll.replace('.h5smu', ''))
        if i == 0:
            im = axes[1].images
            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes('right', size='10%', pad=0.05)
            fig.colorbar(im[-1], cax=cax)
    plt.suptitle(f'{gene_name} expression across samples (aligned with dynamic time-warping)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    pass

##
img_name = 'MAX_20210830_HCR_sample2_C1and9_003.lif - TileScan 001_TileScan_001_Merging.png'
img = np.array(Image.open(os.path.join(root, 'raw/single_molecule', img_name)))
img = np.fliplr(img)
sample_name = 'b3-LWF-17-12hpa'

raster = smu.Raster(X=img)
# plt.imshow(img)
# plt.show()

s = smu.SpatialMuData(os.path.join(root, f'smu/{sample_name}.h5smu'))
x0 = manual_alignments[sample_name][0]
x1 = manual_alignments[sample_name][1]
new_anchor = raster.anchor.map_untransformed_to_untransformed_fov(s['tomo-seq']['image'], source_points=np.array([
    [155, 0],
    [2690, 0]
]), target_points=np.array([
    [x0, 0],
    [x1, 0]
]))
raster._anchor = new_anchor

mask = np.array(Image.open(os.path.join(root, 'raw/single_molecule/mask.png')))
mask[mask != 255] = 0
mask = mask[:, :, 3]
raster_mask = smu.RasterMasks(mask=mask)
regions = smu.Regions(masks=raster_mask, anchor=new_anchor)

s_fish = smu.SpatialMuData(os.path.join(root, 'smu/fish.h5smu'), backingmode='w')
s_fish['fish'] = smu.SpatialModality()
s_fish['fish']['image'] = raster
if 'covered' in s_fish['fish']:
    del s_fish['fish']['covered']
s_fish['fish']['covered'] = regions
s_fish
##
import pandas as pd
df0 = pd.read_csv(os.path.join(root, 'raw/single_molecule/foot_gene_highlight.csv'))
# hard coded, maybe better to have a convenience function to show this from the data
half_slice = 2.5
df0['coord'] = df0.pos.apply(lambda x: x0 + half_slice + (x1 - x0 - 2 * half_slice) * (x - 1) / 57)
df0
##
fig = plt.figure()
ax0, ax1, ax2 = ImageGrid(fig, 111, nrows_ncols=(3, 1), axes_pad=0.1)
ax1.sharex(ax0)
ax2.sharex(ax0)
s["tomo-seq"]["image"].plot(ax=ax0)
# ax0.axvline(x=x0, c='r', linestyle='--')
# ax0.axvline(x=x1, c='r', linestyle='--')
ax0.set_title(sample_name)
s_fish["fish"]["image"].plot(ax=ax1, show_colorbar=False, show_scalebar=False, show_title=False)
x = s_fish['fish']['covered'].masks.data
boolean_mask = x == 255
contours = skimage.measure.find_contours(boolean_mask, 0.7)
for contour in contours:
    contour = np.fliplr(contour)
    transformed = s_fish['fish']['covered'].anchor.transform_coordinates(contour)
    ax1.plot(transformed[:, 0], transformed[:, 1], linewidth=1, color='gray', linestyle='--')
s["tomo-seq"]["expression"].plot(6, ax=ax2, show_colorbar=False)
ax2.plot(df0['coord'].to_numpy(), df0['expr'].to_numpy()[::-1] * 100, '-', c='w')
# ax0.set_xlim((x0, x1))
plt.show()
print('')
pass