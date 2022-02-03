import unittest

import matplotlib.pyplot as plt
import numpy as np

import spatialmuon
from spatialmuon._core.anchor import Anchor
from tests.testing_utils import initialize_testing

test_data_dir, DEBUGGING = initialize_testing()

fpath_ome = test_data_dir / "ome_example.tiff"

# TODO: Anchor is giving me AttributeError: module 'spatialmuon' has no attribute 'Anchor'. I have
#  temporarily changed the import to be more explicit


class Anchor_TestClass(unittest.TestCase):
    def test_can_create_Anchor(self):
        a = Anchor(3)
        self.assertTrue(isinstance(a, Anchor))

    def test_spm_Raster_contains_Anchor(self):
        c = spatialmuon.Converter()
        ome_raster = c.raster_from_tiff(fpath_ome)
        self.assertTrue(isinstance(ome_raster.anchor, Anchor))

    def test_Anchor_contains_ndim(self):
        a = Anchor(3)
        self.assertTrue(hasattr(a, "ndim"))

    def test_Anchor_contains_origin(self):
        a = Anchor(3)
        self.assertTrue(hasattr(a, "origin"))

    def test_Anchor_contains_vector(self):
        a = Anchor(3)
        self.assertTrue(hasattr(a, "vector"))

    def test_move_origin(self):
        a = Anchor(2)
        a.move_origin("x", 2)
        self.assertTrue(np.alltrue(a.origin == np.array([2, 0])))

    def test_rotate_vector(self):
        a = Anchor(2)
        a.rotate_vector(45)
        self.assertEqual([np.round(x, 5) for x in a.vector], [0.70711, 0.70711])

    def test_scale_vector(self):
        a = Anchor(2)
        a.scale_vector(2)
        self.assertTrue(np.alltrue(a.vector == np.array([2, 0])))

    def test_compute_alignment_translation_scale(self):
        ##
        des = np.zeros((100, 100, 3), dtype=np.float)
        des[60:, 80:, :] = np.array([1.0, 0.0, 0.0])
        des_anchor = Anchor(origin=np.array([50, 50]), vector=np.array([2, 0]))
        des_fov = spatialmuon.Raster(X=des, anchor=des_anchor)

        src = np.ones((60, 30, 3), dtype=np.float)
        src[:, :, np.array([0, 2])] = 0.0
        src_anchor = Anchor.map_untransformed_to_untransformed_fov(
            des_fov,
            source_points=np.array([[15, 30], [30, 60]]),
            target_points=np.array([[90, 80], [100, 100]]),
        )
        src_fov = spatialmuon.Raster(X=src, anchor=src_anchor)
        _, (ax0, ax1) = plt.subplots(1, 2)
        des_fov.plot(ax=ax0)
        ax0.set(xlim=(0, 150), ylim=(0, 150))
        src_fov.plot(ax=ax1)
        ax1.set(xlim=(0, 150), ylim=(0, 150))
        ax0.grid()
        ax1.grid()
        plt.show()
        print(src_anchor)
        ##


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main(failfast=True)
    else:
        Anchor_TestClass().test_compute_alignment_translation_scale()
