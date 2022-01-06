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
    fpath = os.path.expanduser("~/spatialmuon/tests/data/small_visium.h5smu")

plt.style.use("dark_background")


##
class PlotSmallVisium_TestClass(unittest.TestCase):
    def test_can_load_smu_file(self):
        spatialmuon.SpatialMuData(backing=fpath)

    def test_can_pretty_print(self):
        ##
        d = spatialmuon.SpatialMuData(backing=fpath)
        print(d)
        pass
        ##

    def test_can_plot_regions_single_channel(self):
        # d = spatialmuon.SpatialMuData(backing=fpath)
        # ome = d["imc"]["ome"]
        # ome.plot(channels="ArAr80", preprocessing=np.arcsinh)
        pass

    def test_can_plot_regions_non_overlapping_channels(self):
        # d = spatialmuon.SpatialMuData(backing=fpath)
        # ome = d["imc"]["ome"]
        # ome.plot(
        #     preprocessing=np.arcsinh,
        # )
        pass

    def test_can_plot_regions_random_color(self):
        # d = spatialmuon.SpatialMuData(backing=fpath)
        # masks = d["imc"]["masks"]
        # masks.plot()
        # masks.masks.plot(fill_colors='random', outline_colors='w')
        # masks.masks.plot(fill_colors=None, outline_colors='random')
        pass

    def test_can_plot_regions_solid_color(self):
        # d = spatialmuon.SpatialMuData(backing=fpath)
        # masks = d["imc"]["masks"]
        # masks.masks.plot(fill_colors='red')
        # masks.masks.plot(fill_colors=[0., 0., 1., 1.])
        # masks.masks.plot(fill_colors=np.array([0., 1., 0., 1.]))
        # colors = ["red", "yellow"] * len(masks.masks.obs)
        # colors = colors[: len(masks.masks.obs)]
        # colors[1] = [0., 1., 0]
        # masks.masks.plot(fill_colors=colors)
        pass

    def test_can_plot_raster_and_regions_together(self):
        # d = spatialmuon.SpatialMuData(backing=fpath)
        # fig, ax = plt.subplots(1)
        # channels = d['imc']['ome'].var['channel_name'].tolist()
        # d['imc']['ome'].plot(ax=ax, channels=channels[0])
        # d['imc']['masks'].masks.plot(fill_colors=None, outline_colors='random', ax=ax)
        # plt.show()
        pass

    def test_can_accumulate_raster_with_shape_masks(self):
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
        PlotSmallVisium_TestClass().test_can_load_smu_file()
        PlotSmallVisium_TestClass().test_can_pretty_print()
        #
        PlotSmallVisium_TestClass().test_can_plot_regions_single_channel()
        PlotSmallVisium_TestClass().test_can_plot_regions_non_overlapping_channels()
        #
        PlotSmallVisium_TestClass().test_can_plot_regions_random_color()
        PlotSmallVisium_TestClass().test_can_plot_regions_solid_color()
        PlotSmallVisium_TestClass().test_can_plot_raster_and_regions_together()
        PlotSmallVisium_TestClass().test_can_accumulate_raster_with_shape_masks()
        PlotSmallVisium_TestClass().test_can_plot_accumulated_regions_value()
