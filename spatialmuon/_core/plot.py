from spatialmuon._core.fieldofview import FieldOfView
import spatialmuon.datatypes
import matplotlib.axes


def spatial(fov: FieldOfView, ax: matplotlib.axes.Axes, **kwargs):
    if type(fov) == spatialmuon.datatypes.raster.Raster:
        spatial_raster(fov, ax, kwargs["channel"])
    else:
        print("aaa")
    # if fov


def spatial_raster(fov: FieldOfView, ax: matplotlib.axes.Axes, channel):
    im = ax.imshow(fov.X[...][:, :, channel], cmap=matplotlib.cm.get_cmap("gray"))
    print("plotting")
    pass

