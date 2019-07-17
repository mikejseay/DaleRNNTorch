import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


class MidpointNormalize(colors.Normalize):
    """ create asymmetric norm """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_weights_square_ax(w, ax, plot_colorbar=True):
    vmin, vmax = np.min(w), np.max(w)
    i = ax.imshow(w, cmap='RdBu', interpolation='none')
    norm = MidpointNormalize(vmin, vmax, 0)
    i.set_norm(norm)
    if plot_colorbar:
        c = plt.colorbar(i, ax=ax)


def plot_weights(w_output, w_input, w_rec):
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
    ax_rec = fig.add_subplot(grid[:-1, 1:], xticklabels=[], yticklabels=[])
    ax_inp = fig.add_subplot(grid[:-1, 0], xticklabels=[], yticklabels=[])
    ax_out = fig.add_subplot(grid[-1, 1:], xticklabels=[], yticklabels=[])

    plot_weights_square_ax(w_input, ax_inp, plot_colorbar=True)
    plot_weights_square_ax(w_rec, ax_rec, plot_colorbar=True)
    plot_weights_square_ax(w_output, ax_out, plot_colorbar=True)


def plot_eigenspectra(w_rec):
    ew, ev = np.linalg.eig(w_rec)

    f, ax = plt.subplots()
    ax.scatter(ew.real, ew.imag)

    print('max real part is', ew.real.max())
    print('dot product of first two eigenvectors is',
          np.dot(ev[:, 0], ev[:, 1]))

    return ew, ev
