import unittest
import spatialmuon
from pathlib import Path
import numpy as np
from spatialmuon._core.anchor import Anchor
import matplotlib.pyplot as plt
from tests.testing_utils import initialize_testing

test_data_dir, DEBUGGING = initialize_testing()

fpath = test_data_dir / "scaled_visium.h5smu"


class Alignment_TestClass(unittest.TestCase):
    def test_plot(self):
        s = spatialmuon.SpatialMuData(fpath)
        s["visium"]["expression"].plot(99)
        s["visium"]["image"].plot()
        s["visium"]["image2x"].plot()
        s["visium"]["image_crop"].plot()
        s["visium"]["image2x_crop"].plot()
        pass


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main()
    else:
        Alignment_TestClass().test_plot()
