import unittest
import spatialmuon
from pathlib import Path
import numpy as np
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


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main()
    else:
        pass
