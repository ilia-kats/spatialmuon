from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from skimage import color


def circle(radius, center):
    theta = np.linspace(0, 2 * np.pi, 200)
    return center + radius * np.exp(1j * theta)


# Get current file and pre-generate paths and names
this_dir = Path(__file__).parent
smily_fpath = this_dir / "ome_example.tiff"

# Draw and save smily
smile = 0.3 * np.exp(1j * 2 * np.pi * np.linspace(0.6, 0.9, 20)) - 0.2j
frown = 0.3 * np.exp(1j * 2 * np.pi * np.linspace(1.1, 1.4, 20)) - 0.6j
left_eye = circle(0.1, -0.4 + 0.2j)
right_eye = circle(0.1, 0.4 + 0.2j)
happy_face = [circle(1, 0), left_eye, smile, right_eye]
sad_face = [circle(1, 0), left_eye, frown, right_eye]

with tiff.TiffWriter(smily_fpath) as tiff_fh:
    for face in [happy_face, sad_face]:

        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.set_aspect(1)
        for idx, shape in enumerate(face):
            ax.plot(shape.real, shape.imag)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = np.reshape(data, (h, w, -1))
        img_grey = color.rgb2gray(img).astype(np.uint16)
        tiff_fh.write(img_grey, photometric="miniswhite")

# generate fake-metadata
metadata = "<ome:OME xmlns:ns2='http://www.openmicroscopy.org/Schemas/BinaryFile/2013-06s' "
metadata += "xmlns:om='openmicroscopy.org/OriginalMetadata' "
metadata += "xmlns:ome='http://www.openmicroscopy.org/Schemas/ome/2013-06s' "
metadata += "xmlns:sa='http://www.openmicroscopy.org/Schemas/sa/2013-06s' "
metadata += "xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance' "
metadata += "xsi:schemaLocation='http://www.openmicroscopy.org/Schemas/OME/2013-06 http://www.openmicroscopy.org/Schemas/OME/2012-03/ome.xsd'>"
metadata += "\n\t<ome:Image ID='Image:0' Name='ome_tiff_smily.tiff'>"
metadata += "\n\t\t<ome:AcquisitionDate>2017-11-28T13:17:31.052385</ome:AcquisitionDate>"
metadata += "\n\t\t<ome:Pixels DimensionOrder='YCZT' ID='Pixels:0' SizeC='2' SizeT='1' SizeX='{}' SizeY='{}' SizeZ='1' Type='float'>".format(
    img_grey.shape[0], img_grey.shape[1]
)
metadata += "\n\t\t\t<ome:Channel Fluor='happy' ID='Channel:0:0' Name='happy' SamplesPerPixel='1'>"
metadata += "\n\t\t\t\t<ome:LightPath />"
metadata += "\n\t\t\t</ome:Channel>"
metadata += "\n\t\t\t<ns2:BinData BigEndian='false' Length='0' />"
metadata += "\n\t\t\t<ome:Channel Fluor='sad' ID='Channel:0:1' Name='sad' SamplesPerPixel='1' /></ome:Pixels>"
metadata += "\n\t\t</ome:Image>"
metadata += "\n\t<sa:StructuredAnnotations>"
metadata += "\n\t\t<sa:XMLAnnotation ID='750ee0e5-1cfc-4313-b41d-11c1ff9feaea'>"
metadata += "\n\t\t\t<sa:Value>"
metadata += "\n\t\t\t\t<om:OriginalMetadata>"
metadata += "\n\t\t\t\t\t<om:Key>MCD-XML</om:Key>"
metadata += "\n\t\t\t\t\t<om:Value />"
metadata += "\n\t\t\t\t</om:OriginalMetadata>"
metadata += "\n\t\t\t</sa:Value>"
metadata += "\n\t\t</sa:XMLAnnotation>"
metadata += "\n\t</sa:StructuredAnnotations>"
metadata += "\n</ome:OME>"
tiff.tiffcomment(smily_fpath, metadata)
