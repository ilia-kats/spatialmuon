import unittest
import spatialmuon
import tempfile
import os
import uuid
import numpy as np
import copy
from tests.testing_utils import initialize_testing

test_data_dir, DEBUGGING = initialize_testing()


class Backing_TestClass(unittest.TestCase):
    # Here we check that the creation of a SpatialMuData object is consistent indipendently of the order in which
    # subobjects (=SpatialModality and FieldOfView) are added to the storage hierarchy, and that when we initialize
    # an object from a subobject taken from another object we are sucessfully resetting the backed storage,
    # so that there are no backed objects with subpieces pointing and operating on other backed objects.
    # Notice that in the cases in which we add a subobject from another object, we can then modify the subobject in the
    # new object since it is not pointing anymore to the old backed storage.
    # Here we are not testing the behavior on all the subobjects, like the various Masks and Anchor objects,
    # dedicated tests could be added, but still, this should generalize decently to classes inheriting from
    # BackableObject and BackedDictProxy
    # flake8: noqa: C901
    def test_various_setitem_orders(self):
        with tempfile.TemporaryDirectory() as td:

            def new_smu(backed: bool):
                if not backed:
                    return spatialmuon.SpatialMuData()
                else:
                    return spatialmuon.SpatialMuData(backing=os.path.join(td, str(uuid.uuid4())))

            def new_smo():
                return spatialmuon.SpatialModality()

            def new_fov():
                return spatialmuon.Raster(X=np.zeros((3, 3, 3)))

            def smf(backed: bool = True):
                s = new_smu(backed=backed)
                m = new_smo()
                f = new_fov()
                return s, m, f

            # string representations
            ss = {}

            def store(case: str):
                nonlocal s
                ss[case] = s

            # for checking that by modifying any fov in any object we are not modifying things in other objects
            def how_many_fovs():
                for s in ss.values():
                    f = s["a"]["a"]
                    f._X = np.random.rand(3, 3, 3)
                reprs = set()
                for s in ss.values():
                    f = s["a"]["a"]
                    reprs.add(str(f._X))
                return len(reprs)

            # below we are not taking care of listing all the possible cases, but to cover a reasonable portion of them

            # --- cases where all objects are created from scratch ---
            # case a: backed storage, non-backed mods and fovs; two different orders
            s, m, f = smf()
            s["a"] = m
            m["a"] = f
            store("a0")
            assert how_many_fovs() == 1

            s, m, f = smf()
            m["a"] = f
            s["a"] = m
            store("a1")
            assert how_many_fovs() == 2

            # case b: as above, but non-backed storage
            s, m, f = smf(backed=False)
            s["a"] = m
            m["a"] = f
            store("b0")
            # print(id(f))
            assert how_many_fovs() == 3

            s, m, f = smf(backed=False)
            m["a"] = f
            s["a"] = m
            store("b1")
            assert how_many_fovs() == 4

            # --- cases where fovs are borrowed from other objects ---
            # case c: as case a, but fov is (a backed storage) coming from case a0
            s, m, _ = smf()
            s["a"] = m
            m["a"] = ss["a0"]["a"]["a"]
            store("c0")
            assert how_many_fovs() == 5

            s, m, _ = smf()
            m["a"] = ss["a0"]["a"]["a"]
            s["a"] = m
            store("c1")
            assert how_many_fovs() == 6

            # case d: as case a, but fov is (a backed storage) coming from case a1
            s, m, _ = smf()
            s["a"] = m
            m["a"] = ss["a1"]["a"]["a"]
            store("d0")
            assert how_many_fovs() == 7

            s, m, _ = smf()
            m["a"] = ss["a1"]["a"]["a"]
            s["a"] = m
            store("d1")
            assert how_many_fovs() == 8

            # case e: as case a, but fov is (a non-backed storage) coming from case b0
            # here we need to copy the inner object as there is no current way for spatial muon to know if a fov
            # being copied belongs to another object when the copied object is not backed
            s, m, _ = smf()
            s["a"] = m
            m["a"] = copy.copy(ss["b0"]["a"]["a"])
            # m["a"] = ss["b0"]["a"]["a"].clone()
            store("e0")
            # print(id(s["a"]["a"]))
            # print(id(ss["b0"]["a"]["a"]))
            assert how_many_fovs() == 9

            s, m, _ = smf()
            m["a"] = copy.copy(ss["b0"]["a"]["a"])
            # m["a"] = ss["b0"]["a"]["a"].clone()
            s["a"] = m
            store("e1")
            assert how_many_fovs() == 10
            # case f: as case a, but fov is (a non-backed storage) coming from case b1
            # here we need to copy the inner object as there is no current way for spatial muon to know if a fov
            # being copied belongs to another object when the copied object is not backed
            s, m, _ = smf()
            s["a"] = m
            m["a"] = copy.copy(ss["b1"]["a"]["a"])
            # m["a"] = ss["b1"]["a"]["a"].clone()
            store("f0")
            assert how_many_fovs() == 11

            s, m, _ = smf()
            m["a"] = copy.copy(ss["b1"]["a"]["a"])
            # m["a"] = ss["b1"]["a"]["a"].clone()
            s["a"] = m
            store("f1")
            assert how_many_fovs() == 12

            # --- cases where both mods and fovs are borrowed from other objects ---
            # case g: as case a, but not both mod and fov are borrowed, respectively, from case f0 and f1,
            s, _, _ = smf()
            s["a"] = ss["f0"]["a"]
            store("g0")
            assert how_many_fovs() == 13
            s, _, _ = smf()
            s["a"] = ss["f1"]["a"]
            store("g1")
            assert how_many_fovs() == 14
            # case h: as case b, but not both mod and fov are borrowed, respectively, from case g0 and g1,
            s, _, _ = smf()
            s["a"] = ss["g0"]["a"]
            store("h0")
            assert how_many_fovs() == 15
            s, _, _ = smf()
            s["a"] = ss["g1"]["a"]
            store("h1")
            assert how_many_fovs() == 16
            # TODO: add tests: mod borrowed from a non-backed object
            # TODO: add tests for masks and anchors (in another test function)
            # TODO: add tests that currently make the other tests fail
            ##
            # check that all the objects contain the same things
            reprs = set()
            for s in ss.values():
                reprs.add(str(s))
            assert len(reprs) == 1
            ##
            assert how_many_fovs() == len(ss)
            ##
            # check that creating a fov for each mod does not create fovs in other mods that were borrowed from
            # other objects
            for s in ss.values():
                m = s["a"]
                f = new_fov()
                m[str(uuid.uuid4())] = f

            for s in ss.values():
                assert len(s["a"].keys()) == 2
            ##

    @staticmethod
    def _create_regions_in_dir(directory):
        masks_centers = np.array([[10, 10]])
        masks_radii = np.array([[1, 1]])
        fpath = os.path.join(directory, "target.h5smu")
        s = spatialmuon.SpatialMuData(backing=fpath)
        m = spatialmuon.SpatialModality()
        sm = spatialmuon.ShapeMasks(masks_centers=masks_centers, masks_radii=masks_radii)
        f = spatialmuon.Regions(masks=sm)
        m["a"] = f
        s["a"] = m
        return s

    @staticmethod
    def _shape_masks_differ(s0, s1):
        x0 = s0["a"]["a"].masks.untransformed_masks_centers[...]
        x1 = s1["a"]["a"].masks.untransformed_masks_centers[...]
        b = np.any(x0 != x1)
        return b

    def test_cloning_to_file(self):
        print("test_cloning_to_file")
        with tempfile.TemporaryDirectory() as td:
            s0 = self._create_regions_in_dir(directory=td)
            f = os.path.join(td, "clone.h5smu")
            s1 = s0.clone_to_file(f)
            shape_masks = spatialmuon.ShapeMasks(
                masks_centers=np.array([[9, 9]]), masks_radii=np.array([[1, 1]])
            )
            del s1["a"]["a"]["masks"]
            s1["a"]["a"].masks = shape_masks
            assert self._shape_masks_differ(s0, s1)

    def test_on_demand_save(self):
        print("test_on_demand_save")
        with tempfile.TemporaryDirectory() as td:
            s0 = self._create_regions_in_dir(directory=td)
            f = os.path.join(td, "clone.h5smu")
            s1 = s0.clone_to_file(f)

            assert not self._shape_masks_differ(s0, s1)
            x = np.array([[9, 9]])
            s1["a"]["a"].masks.untransformed_masks_centers = x
            assert self._shape_masks_differ(s0, s1)

            s2 = spatialmuon.SpatialMuData(s1.backing.filename, backingmode="r")
            assert self._shape_masks_differ(s1, s2)

            s1.commit_changes_on_disk()
            s3 = spatialmuon.SpatialMuData(s1.backing.filename, backingmode="r")
            assert not self._shape_masks_differ(s1, s3)

    def test_inplace_operations(self):
        print("test_inplace_operations")
        with tempfile.TemporaryDirectory() as td:
            s0 = self._create_regions_in_dir(directory=td)
            f = os.path.join(td, "clone.h5smu")
            s1 = s0.clone_to_file(f)

            assert not self._shape_masks_differ(s0, s1)
            x = np.array([[9, 9]])
            t = s1["a"]["a"].masks.untransformed_masks_centers
            t[...] = x
            assert self._shape_masks_differ(s0, s1)

            s2 = spatialmuon.SpatialMuData(s1.backing.filename, backingmode="r")
            assert not self._shape_masks_differ(s1, s2)

    def test_copy_is_not_shallow(self):
        with tempfile.TemporaryDirectory() as td:
            s0 = self._create_regions_in_dir(directory=td)
            f = os.path.join(td, "clone.h5smu")
            s1 = spatialmuon.SpatialMuData(backing=f)
            s1["a"] = s0["a"]
            t = s0["a"]["a"].masks.untransformed_masks_centers
            x = np.array([[9, 9]])
            t[...] = x
            s2 = spatialmuon.SpatialMuData(backing=f)
            assert self._shape_masks_differ(s0, s2)
            assert self._shape_masks_differ(s0, s1)


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main(failfast=True)
    else:
        # Backing_TestClass().test_various_setitem_orders()
        # Backing_TestClass().test_cloning_to_file()
        # Backing_TestClass().test_on_demand_save()
        Backing_TestClass().test_inplace_operations()
        # Backing_TestClass().test_copy_is_not_shallow()
