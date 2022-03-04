import tempfile
import shutil
import os
import unittest
import copy
import numpy as np
import pandas as pd

import spatialmuon
from spatialmuon.processing.tiler import Tiles
from tests.testing_utils import initialize_testing

test_data_dir, DEBUGGING = initialize_testing()

fpath_imc = test_data_dir / "small_imc.h5smu"


class VarObs_TestClass(unittest.TestCase):
    def test_can_save_and_load_obs(self):
        ##
        with tempfile.TemporaryDirectory() as td:
            ##
            des = os.path.join(td, "small_imc.h5smu")
            shutil.copy(fpath_imc, des)
            s = spatialmuon.SpatialMuData(backing=des)

            regions = s["imc"]["masks"]
            regions.masks.obs["new_col"] = np.array(range(22)) + 1000
            cat = pd.Series(["a" for _ in range(22)]).astype("category")
            regions.masks.obs["cat"] = cat
            print(regions.masks.obs)
            regions.set_all_has_changed(new_value=True)
            s.commit_changes_on_disk()
            ##
            t = spatialmuon.SpatialMuData(s.backing.filename)
            m = t["imc"]["masks"].masks
            print(m.obs)
            print('ooo')
            ##
            non_backed_regions = regions.clone()
            s['imc']['masks2'] = non_backed_regions
            s.commit_changes_on_disk()
            ##
            print(dict(s['imc']['masks'].masks.backing['obs']['cat'].attrs))
            t = spatialmuon.SpatialMuData(s.backing.filename)
            print(dict(t['imc']['masks'].masks.backing['obs']['cat'].attrs))

            # weird bug. workaround: call .clone() for masks instead of passing a backed one instead the non-backed
            # Regions
            # describption:
            # if we create a Regions with inside a masks that is taken from a backed file, and that masks has inside
            # a categorical column, then if we save the regions to a file (by putting it in a backed muon object)
            # then the following dict dict(smu['modality']['regions'].masks.backing['obs'][
            # 'categorical_column'].attrs) will contain a h5py.h5r.Reference object that is not valid,
            # probably because inside backing.py in _clone_at_current_level() we don't deal with that datatype
            import h5py.h5r
            h5py.h5r.Reference
            ##


if __name__ == "__main__":
    if not DEBUGGING:
        unittest.main(failfast=True)
    else:
        VarObs_TestClass().test_can_save_and_load_obs()
