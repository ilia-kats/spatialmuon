import unittest
import spatialmuon
import tempfile
import os
import uuid
import numpy as np
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

            # below we are not taking care of listing all the possible cases, but to cover a reasonable portion of them

            # --- cases where all objects are created from scratch ---
            # case a: backed storage, non-backed mods and fovs; two different orders
            s, m, f = smf()
            s["a"] = m
            m["a"] = f
            store("a0")

            s, m, f = smf()
            m["a"] = f
            s["a"] = m
            store("a1")

            # case b: as above, but non-backed storage
            s, m, f = smf(backed=False)
            s["a"] = m
            m["a"] = f
            store("b0")

            s, m, f = smf(backed=False)
            m["a"] = f
            s["a"] = m
            store("b1")

            # --- cases where fovs are borrowed from other objects ---
            # case c: as case a, but fov is (a backed storage) coming from case a1
            s, m, _ = smf()
            s["a"] = m
            m["a"] = ss["a0"]['a']['a']
            store("c0")

            s, m, _ = smf()
            m["a"] = ss["a0"]['a']['a']
            s["a"] = m
            store("c1")

            # case d: as case a, but fov is (a backed storage) coming from case a2
            s, m, _ = smf()
            s["a"] = m
            m["a"] = ss["a1"]['a']['a']
            store("d0")

            s, m, _ = smf()
            m["a"] = ss["a1"]['a']['a']
            s["a"] = m
            store("d1")

            # case e: as case a, but fov is (a non-backed storage) coming from case b1
            s, m, _ = smf()
            s["a"] = m
            m["a"] = ss["b0"]['a']['a']
            store("e0")

            s, m, _ = smf()
            m["a"] = ss["b0"]['a']['a']
            s["a"] = m
            store("e1")

            # case f: as case a, but fov is (a non-backed storage) coming from case b2
            s, m, _ = smf()
            s["a"] = m
            m["a"] = ss["b1"]['a']['a']
            store("f0")

            s, m, _ = smf()
            m["a"] = ss["b1"]['a']['a']
            s["a"] = m
            store("f1")

            # --- cases where both mods and fovs are borrowed from other objects ---
            # case g: as case a, but not both mod and fov are borrowed, respectively, from case f0 and f1,
            s, _, _ = smf()
            s["a"] = ss["f0"]['a']
            store("g0")

            s, _, _ = smf()
            s["a"] = ss["f1"]['a']
            store("g1")

            # case h: as case b, but not both mod and fov are borrowed, respectively, from case g0 and g1,
            s, _, _ = smf()
            s["a"] = ss["g0"]['a']
            store("h0")

            s, _, _ = smf()
            s["a"] = ss["g1"]['a']
            store("h1")
            ##
            # check that all the objects contain the same things
            reprs = set()
            for s in ss.values():
                reprs.add(str(s))
            assert len(reprs) == 1
            ##
            # check that by modifying any fov in any object we are not modifying things in other objects
            for s in ss.values():
                f = s['a']['a']
                f._X = np.random.rand(3, 3, 3)
            reprs = set()
            for s in ss.values():
                f = s['a']['a']
                reprs.add(str(f._X))
            assert len(reprs) == len(ss)
            ##
            # check that creating a fov for each mod does not create fovs in other mods that were borrowed from
            # other objects
            for s in ss.values():
                m = s['a']
                f = new_fov()
                m[str(uuid.uuid4())] = f

            for s in ss.values():
                assert len(s['a'].keys()) == 2
            ##
            print("ooooo")
            print("ooooo")


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main()
    else:
        Backing_TestClass().test_various_setitem_orders()
