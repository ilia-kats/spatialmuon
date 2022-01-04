##
import unittest
import sys
import spatialmuon
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DEBUGGING = False
try:
    __file__
except NameError as e:
    if str(e) == "name '__file__' is not defined":
        IN_PYCHARM = True
    else:
        raise e
if sys.gettrace() is not None:
    DEBUGGING = True

if not DEBUGGING:
    # Get current file and pre-generate paths and names
    this_dir = Path(__file__).parent
    fpath = this_dir / "../data/small_imc.h5smu"

    matplotlib.use("Agg")
else:
    fpath = "/data/l989o/deployed/spatialmuon/tests/data/small_imc.h5smu"

plt.style.use("dark_background")


##
class Converter_TestClass(unittest.TestCase):
    def test_can_load_smu_file(self):
        spatialmuon.SpatialMuData(backing=fpath)

    def test_can_pretty_print(self):
        ##
        d = spatialmuon.SpatialMuData(backing=fpath)
        print(d)
        ##

    def test_can_plot_single_channel_in_ax(self):
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

    def test_can_plot_overlapping_channels_in_ax(self):
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

    def test_can_plot_single_channel(self):
        ##
        d = spatialmuon.SpatialMuData(backing=fpath)
        ome = d["imc"]["ome"]
        ome.plot(channels="ArAr80", preprocessing=np.arcsinh)
        ##

    def test_can_plot_overlapping_channels(self):
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
            overlap=True,
        )
        ##

    def test_can_plot_non_overlapping_channels(self):
        d = spatialmuon.SpatialMuData(backing=fpath)
        ome = d["imc"]["ome"]
        ome.plot(
            preprocessing=np.arcsinh,
        )


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main()
    else:
        # Converter_TestClass().test_can_plot_single_channel_in_ax()
        Converter_TestClass().test_can_plot_single_channel()
        # Converter_TestClass().test_can_plot_overlapping_channels_in_ax()
        # Converter_TestClass().test_can_plot_overlapping_channels()
        Converter_TestClass().test_can_plot_non_overlapping_channels()
