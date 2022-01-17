##
import unittest
import os
import sys
import spatialmuon
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import copy
from pathlib import Path

DEBUGGING = False
try:
    __file__
except NameError as e:
    if str(e) == "name '__file__' is not defined":
        DEBUGGING = True
    else:
        raise e
if sys.gettrace() is not None:
    DEBUGGING = True

if not DEBUGGING:
    # Get current file and pre-generate paths and names
    this_dir = Path(__file__).parent
    fpath = this_dir / "../data/small_visium.h5smu"

    matplotlib.use("Agg")
else:
    small_visium = "~/spatialmuon/tests/data/small_visium.h5smu"
    fpath = os.path.expanduser(small_visium)

plt.style.use("dark_background")


##
class PlotSmallVisium_TestClass(unittest.TestCase):
    def test_can_load_smu_file(self):
        spatialmuon.SpatialMuData(backing=fpath)

    def test_can_pretty_print(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        print(d)

    def test_can_plot_image(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        img = d["visium"]["image"]
        img.plot()

    def test_can_plot_regions_single_channel(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        e = d["visium"]["expression"]
        plt.figure()
        ax = plt.gca()
        e.plot(channels="Rp1", ax=ax)
        ax.set(xlim=(1650, 1700), ylim=(1500, 1600))
        plt.show()

    def test_can_plot_regions_non_overlapping_channels(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        e = d["visium"]["expression"]
        e.plot(list(range(10)))

    def test_can_plot_regions_random_color(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        e = d["visium"]["expression"]
        _, ax = plt.subplots(1)
        e.masks.plot(fill_colors="black", outline_colors="random", ax=ax)
        ax.set_title("visualizing masks")
        plt.show()

    def test_can_plot_regions_solid_color(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        e = d["visium"]["expression"]
        e.masks.plot(fill_colors="red")
        e.masks.plot(fill_colors=[0.0, 0.0, 1.0, 1.0])
        e.masks.plot(fill_colors=np.array([0.0, 1.0, 0.0, 1.0]))
        colors = ["red", "yellow"] * len(e.masks.obs)
        colors = colors[: len(e.masks.obs)]
        colors[1] = [0.0, 1.0, 0]
        e.masks.plot(fill_colors=colors)

    def test_bounding_boxes(self):
        ##
        d = spatialmuon.SpatialMuData(backing=fpath)
        e = d["visium"]["expression"]
        im = d["visium"]["image"]
        bb_e = e.bounding_box
        bb_im = im.bounding_box
        # print(bb_e, bb_im)

    ##

    def test_can_plot_raster_and_regions_together(self):
        ##
        d = spatialmuon.SpatialMuData(backing=fpath)
        e = d["visium"]["expression"]
        _, ax = plt.subplots(1, figsize=(5, 5))
        d["visium"]["image"].plot(ax=ax, show_legend=False)
        # e.plot(channels=6, ax=ax, )
        e.masks.plot(fill_colors=None, outline_colors='red', ax=ax)
        ax.set_title("visium spots")
        e.set_lims_to_bounding_box()
        # ax.set(xlim=(1400, 1500), ylim=(1600, 2000))
        plt.show()
        ##

    def test_can_accumulate_raster_with_shape_masks(self):
        # raise NotImplementedError()
        # d = spatialmuon.SpatialMuData(backing=fpath)
        # accumulated = d['imc']['ome'].accumulate_features(d['imc']['masks'].masks)
        # for k, v in accumulated.items():
        #     if k in d['imc']:
        #         del d['imc'][k]
        #     d['imc'][k] = v
        # for k in accumulated.keys():
        #     del d['imc'][k]
        pass

    def test_can_plot_accumulated_regions_value(self):
        # raise NotImplementedError()
        # d = spatialmuon.SpatialMuData(backing=fpath)
        # accumulated = d['imc']['ome'].accumulate_features(d['imc']['masks'].masks)
        # for k, v in accumulated.items():
        #     if k in d['imc']:
        #         del d['imc'][k]
        #     d['imc'][k] = v
        # feature = 'mean'
        # fig, ax = plt.subplots(1)
        # d['imc'][feature].plot(channels=0, preprocessing=np.arcsinh, suptitle=feature, ax=ax)
        # # d['imc']['masks'].masks.plot(fill_colors=None, outline_colors='k', ax=ax)
        # plt.show()
        # for k in accumulated.keys():
        #     del d['imc'][k]
        pass


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main()
    else:
        # PlotSmallVisium_TestClass().test_can_load_smu_file()
        # PlotSmallVisium_TestClass().test_can_pretty_print()
        #
        # PlotSmallVisium_TestClass().test_can_plot_image()
        #
        # PlotSmallVisium_TestClass().test_can_plot_regions_single_channel()
        # PlotSmallVisium_TestClass().test_can_plot_regions_non_overlapping_channels()
        #
        # PlotSmallVisium_TestClass().test_can_plot_regions_random_color()
        # PlotSmallVisium_TestClass().test_can_plot_regions_solid_color()
        # PlotSmallVisium_TestClass().test_bounding_boxes()
        PlotSmallVisium_TestClass().test_can_plot_raster_and_regions_together()

        # not implemented yet
        # PlotSmallVisium_TestClass().test_can_accumulate_raster_with_shape_masks()
        # PlotSmallVisium_TestClass().test_can_plot_accumulated_regions_value()
