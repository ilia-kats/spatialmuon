##
import unittest
import spatialmuon
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from tests.testing_utils import initialize_testing

test_data_dir, DEBUGGING = initialize_testing()

fpath = test_data_dir / "small_imc.h5smu"

plt.style.use("dark_background")


##
class PlotSmallImc_TestClass(unittest.TestCase):
    def test_can_load_smu_file(self):
        spatialmuon.SpatialMuData(backing=fpath)

    def test_can_pretty_print(self):
        ##
        d = spatialmuon.SpatialMuData(backing=fpath)
        print(d)
        ##

    def test_can_plot_raster_single_channel_in_ax(self):
        ##
        plt.figure()
        ax = plt.gca()
        d = spatialmuon.SpatialMuData(backing=fpath)
        ome = d["imc"]["ome"]
        ome.plot(ax=ax, channels="ArAr80", preprocessing=np.arcsinh)
        plt.tight_layout()
        plt.subplots_adjust()
        plt.show()
        ##

    def test_can_plot_raster_overlapping_channels_in_ax(self):
        ##
        red_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "red"])
        yellow_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "yellow"])
        blue_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "blue"])

        plt.figure()
        ax = plt.gca()
        d = spatialmuon.SpatialMuData(backing=fpath)
        ome = d["imc"]["ome"]
        ome.plot(
            ax=ax,
            channels=["ArAr80", "Ru96", "Ru98"],
            cmap=[red_cmap, yellow_cmap, blue_cmap],
            preprocessing=np.arcsinh,
        )
        plt.tight_layout()
        plt.show()
        ##

    def test_can_plot_raster_single_channel(self):
        ##
        d = spatialmuon.SpatialMuData(backing=fpath)
        ome = d["imc"]["ome"]
        ome.plot(channels="ArAr80", preprocessing=np.arcsinh)
        ##

    def test_can_plot_raster_overlapping_channels(self):
        ##
        red_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "red"])
        yellow_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "yellow"])
        blue_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "blue"])

        d = spatialmuon.SpatialMuData(backing=fpath)
        ome = d["imc"]["ome"]
        ome.plot(
            channels=["ArAr80", "Ru96", "Ru98"],
            cmap=[red_cmap, yellow_cmap, blue_cmap],
            preprocessing=np.arcsinh,
            method="overlap",
        )
        ##

    def test_can_plot_raster_non_overlapping_channels(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        ome = d["imc"]["ome"]
        ome.plot(
            preprocessing=np.arcsinh,
        )

    def test_can_plot_raster_first_4_channels_as_rgba(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        ome = d["imc"]["ome"]
        ome.plot(
            channels=["ArAr80", "Ru96", "Ru98", "In115"], preprocessing=np.arcsinh, method="rgba"
        )

    def test_can_plot_regions_random_color(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        masks = d["imc"]["masks"]
        masks.plot()
        masks.masks.plot(fill_colors="random", outline_colors="w")
        masks.masks.plot(fill_colors=None, outline_colors="random")

    def test_can_plot_regions_solid_color(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        masks = d["imc"]["masks"]
        masks.masks.plot(fill_colors="red")
        masks.masks.plot(fill_colors=[0.0, 0.0, 1.0, 1.0])
        masks.masks.plot(fill_colors=np.array([0.0, 1.0, 0.0, 1.0]))
        colors = ["red", "yellow"] * len(masks.masks.obs)
        colors = colors[: len(masks.masks.obs)]
        colors[1] = [0.0, 1.0, 0]
        masks.masks.plot(fill_colors=colors)

    def test_can_plot_raster_and_regions_together(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        fig, ax = plt.subplots(1)
        channels = d["imc"]["ome"].var["channel_name"].tolist()
        d["imc"]["ome"].plot(ax=ax, channels=channels[0])
        d["imc"]["masks"].masks.plot(fill_colors=None, outline_colors="random", ax=ax)
        plt.show()

    def test_can_accumulate_raster_with_raster_masks(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        accumulated = d["imc"]["ome"].accumulate_features(d["imc"]["masks"].masks)
        for k, v in accumulated.items():
            if k in d["imc"]:
                del d["imc"][k]
            d["imc"][k] = v
        for k in accumulated.keys():
            del d["imc"][k]

    def test_can_plot_accumulated_regions_value(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        accumulated = d["imc"]["ome"].accumulate_features(d["imc"]["masks"].masks)
        for k, v in accumulated.items():
            if k in d["imc"]:
                del d["imc"][k]
            d["imc"][k] = v
        feature = "mean"
        fig, ax = plt.subplots(1)
        d["imc"][feature].plot(channels=0, preprocessing=np.arcsinh, suptitle=feature, ax=ax)
        d["imc"]["masks"].masks.plot(fill_colors=None, outline_colors="k", ax=ax)
        plt.show()
        for k in accumulated.keys():
            del d["imc"][k]

    def test_can_plot_accumulated_regions_first_4_channels_as_rgba(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        accumulated = d["imc"]["ome"].accumulate_features(d["imc"]["masks"].masks)
        for k, v in accumulated.items():
            if k in d["imc"]:
                del d["imc"][k]
            d["imc"][k] = v
        feature = "mean"
        fig, ax = plt.subplots(1)
        d["imc"][feature].plot(
            channels=[0, 1, 2, 3], preprocessing=np.arcsinh, suptitle=feature, ax=ax, method="rgba"
        )
        plt.show()
        for k in accumulated.keys():
            del d["imc"][k]

    def test_can_plot_accumulated_regions_non_overlapping_channels(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        accumulated = d["imc"]["ome"].accumulate_features(d["imc"]["masks"].masks)
        for k, v in accumulated.items():
            if k in d["imc"]:
                del d["imc"][k]
            d["imc"][k] = v
        feature = "mean"
        d["imc"][feature].plot(preprocessing=np.arcsinh, suptitle=feature)
        for k in accumulated.keys():
            del d["imc"][k]

    def test_bounding_boxes(self):
        ##
        d = spatialmuon.SpatialMuData(backing=fpath)
        bb_ome = d["imc"]["ome"].bounding_box
        bb_mean = d["imc"]["masks"].bounding_box
        # fig, ax = plt.subplots(1)
        # d["imc"][feature].plot(preprocessing=np.arcsinh, suptitle=feature, ax=ax)
        # plt.show()
        ##


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main(failfast=True)
    else:
        # PlotSmallImc_TestClass().test_can_load_smu_file()
        # PlotSmallImc_TestClass().test_can_pretty_print()
        # #
        # PlotSmallImc_TestClass().test_can_plot_raster_single_channel_in_ax()
        # PlotSmallImc_TestClass().test_can_plot_raster_single_channel()
        # PlotSmallImc_TestClass().test_can_plot_raster_overlapping_channels_in_ax()
        # PlotSmallImc_TestClass().test_can_plot_raster_overlapping_channels()
        # PlotSmallImc_TestClass().test_can_plot_raster_non_overlapping_channels()
        # PlotSmallImc_TestClass().test_can_plot_raster_first_4_channels_as_rgba()
        # #
        # PlotSmallImc_TestClass().test_can_plot_regions_random_color()
        # PlotSmallImc_TestClass().test_can_plot_regions_solid_color()
        # PlotSmallImc_TestClass().test_can_plot_raster_and_regions_together()
        # #
        # PlotSmallImc_TestClass().test_can_accumulate_raster_with_raster_masks()
        PlotSmallImc_TestClass().test_can_plot_accumulated_regions_value()
        # PlotSmallImc_TestClass().test_can_plot_accumulated_regions_first_4_channels_as_rgba()
        # PlotSmallImc_TestClass().test_can_plot_accumulated_regions_non_overlapping_channels()
        # #
        # PlotSmallImc_TestClass().test_bounding_boxes()
