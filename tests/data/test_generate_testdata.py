import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Get current file and pre-generate paths and names
this_dir = Path(__file__).parent
smily_fpath = this_dir / "smily.tiff"

# Calculate and draw smily
def circle(radius, center):
    theta = np.linspace(0, 2 * np.pi, 200)
    return center + radius * np.exp(1j * theta)


def plot_curves(curves):
    for c in curves:
        plt.plot(c.real, c.imag)
    plt.axes().set_aspect(1)
    plt.show()
    plt.close()

dash = np.linspace(0.60, 0.90, 20)
smile = 0.3 * np.exp(1j * 2 * np.pi * dash) - 0.2j
left_eye  = circle(0.1, -0.4 + 0.2j)
right_eye = circle(0.1,  0.4 + 0.2j)
face = [circle(1, 0), left_eye, smile, right_eye]

fig, ax = plt.subplots()

for shape in face:
    ax.plot(shape.real, shape.imag)
ax.set_aspect(1)
ax.axis("off")

fig.savefig(smily_fpath)