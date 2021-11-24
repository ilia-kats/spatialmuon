import pytest
import unittest

import numpy as np
import pandas as pd
import spatialmuon as sm

@pytest.mark.usefixtures("filepath_h5smu")
class TestSimpleSpatialMuData:
    def test_mudata_creation(self, filepath_h5smu):
        # Create a small demo dataset
        np.random.seed(1000)
        N, D = 100, 20
        X = np.random.normal(size=(N, D))
        obs = pd.DataFrame(index=[f"obs_{i}" for i in range(N)])
        var = pd.DataFrame(index=[f"var_{i}" for i in range(D)])
        
        coords = np.abs(np.random.normal(size=(N, 2)))

        fovname = "myfov"
        fovidx = 0

        radius = 1.0

        smudata = sm.SpatialMuData(filepath_h5smu)
        smudata["Visium"] = modality = sm.SpatialModality(coordinate_unit="px")

        spots_dict = {o: ((x, y), radius) for (o, (x, y)) in zip(obs.index.tolist(), coords)}
        masks = sm.ShapeMasks(masks_dict=spots_dict, obs=obs)
        cfov = sm.Regions(
            X=X,
            var=var,
            translation=[0, 0, fovidx * 10],
            scale=1.23,
            masks=masks
        )
        modality[fovname] = cfov

        assert smudata["Visium"][fovname].n_var == D
        assert smudata["Visium"][fovname]._masks.n_obs == N


if __name__ == "__main__":
    unittest.main()
