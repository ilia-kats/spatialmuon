import numpy as np
from sklearn.decomposition import PCA
from .fieldofview import FieldOfView
from .spatialmodality import SpatialModality
import spatialmuon.datatypes


def pca(sm: SpatialModality, n_components: int):
    if len(sm) == 0:
        print("modality does not contain field of views")
        return
    elif len(sm) > 1:
        print("currently only modalities with 1 field of views can be plotted")
        return
    else:
        fov = sm.values().__iter__().__next__()
    if (
        type(fov) == spatialmuon.datatypes.raster.Raster
        or type(fov) == spatialmuon.datatypes.array.Array
    ):
        pca = pca_raster_array(fov, n_components)

    else:
        print("aaa")


def pca_raster_array(fov: FieldOfView, n_components: int):
    x = fov.X[...]
    reducer = PCA(n_components)
    pca = reducer.fit_transform(x)
    return pca
