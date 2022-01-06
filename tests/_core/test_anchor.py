import unittest
import spatialmuon
from pathlib import Path

# Get current file and pre-generate paths and names
this_dir = Path(__file__).parent
fpath_ome = this_dir / "../data/ome_example.tiff"


class Anchor_TestClass(unittest.TestCase):
    def test_can_create_Anchor(self):
        a = spatialmuon.Anchor(3)
        self.assertTrue(isinstance(a, spatialmuon.Anchor))

    def test_spm_Raster_contains_Anchor(self):
        c = spatialmuon.Converter()
        ome_raster = c.raster_from_tiff(fpath_ome)
        self.assertTrue(isinstance(ome_raster.anchor, spatialmuon.Anchor))
        
    def test_Anchor_contains_ndim(self):
        a = spatialmuon.Anchor(3)
        self.assertTrue(hasattr(a, "ndim"))
        
    def test_Anchor_contains_origin(self):
        a = spatialmuon.Anchor(3)
        self.assertTrue(hasattr(a, "origin"))
        
    def test_Anchor_contains_vector(self):
        a = spatialmuon.Anchor(3)
        self.assertTrue(hasattr(a, "vector"))
        
    def test_move_origin(self):
        a = spatialmuon.Anchor(2)
        a.move_origin("x", 2)
        self.assertEual(a.origin, np.array([2, 0]))
        
    def test_rotate_vector(self):
        a = spatialmuon.Anchor(2)
        a.rotate_vector(45)
        self.assertEual(
            [np.round(x, 5) for x in a.vector], 
            [0.70711, 0.70711]
        )
        
    def test_scale_vector(self):
        a = spatialmuon.Anchor(2)
        a.scale_vector(2)
        self.assertEual(a.vector, np.array([2, 0]))


if __name__ == "__main__":
    unittest.main()
