import unittest

from tests.data.get_data import get_small_scaled_visium
from tests.testing_utils import initialize_testing

test_data_dir, DEBUGGING = initialize_testing()


class Alignment_TestClass(unittest.TestCase):
    def test_plot(self):
        s = get_small_scaled_visium()
        s["visium"]["expression"].plot(99)
        s["visium"]["image"].plot()
        s["visium"]["image2x"].plot()
        s["visium"]["image_crop"].plot()
        s["visium"]["image2x_crop"].plot()
        import spatialmuon
        s2 = spatialmuon.SpatialMuData()
        s2['visium'] = spatialmuon.SpatialModality()
        s2['visium2'] = spatialmuon.SpatialModality()
        del s2['visium']
        del s2['visium2']
        pass


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main(failfast=True)
    else:
        Alignment_TestClass().test_plot()
