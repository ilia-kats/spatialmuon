import os

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from matplotlib.patches import Rectangle
from skimage import color


def circle(radius, center):
    theta = np.linspace(0, 2 * np.pi, 200)
    return center + radius * np.exp(1j * theta)


plt.style.use("dark_background")

# Get current file and pre-generate paths and names
smily_fpath = "/home/treis/spatialmuon/tests/data/"
fnames = ["mask_left_eye.tiff", "mask_right_eye.tiff", "mask_mouth.tiff"]
rects = [
    Rectangle((0.2, 0), 0.4, 0.4, facecolor="white"),  # left eye
    Rectangle((-0.6, 0), 0.4, 0.4, facecolor="white"),  # right eye
    Rectangle((-0.4, -0.55), 0.8, 0.4, facecolor="white"),  # mouth
]

# Draw smily used in ome_example to ensure same size
smile = 0.3 * np.exp(1j * 2 * np.pi * np.linspace(0.6, 0.9, 20)) - 0.2j
right_eye = circle(0.1, -0.4 + 0.2j)
left_eye = circle(0.1, 0.4 + 0.2j)
happy_face = [circle(1, 0), left_eye, right_eye, smile]


for i in range(0, 3):
    path = os.path.join(smily_fpath, fnames[i])
    with tiff.TiffWriter(path) as tiff_fh:

        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.set_aspect(1)
        ax.plot(happy_face[0].real, happy_face[0].imag, c="black")
        ax.plot(happy_face[i + 1].real, happy_face[i + 1].imag, c="white")
        ax.add_patch(rects[i])
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = np.reshape(data, (h, w, -1))
        img_grey = color.rgb2gray(img).astype(np.uint16)
        tiff_fh.write(img_grey, photometric="minisblack")
