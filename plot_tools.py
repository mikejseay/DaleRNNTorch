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
    else:
        c = None

    return i, c


def plot_weights(w_output, w_input, w_rec):
    fig = plt.figure(figsize=(10, 8))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
    ax_rec = fig.add_subplot(grid[:-1, 1:], xticklabels=[], yticklabels=[])
    ax_inp = fig.add_subplot(grid[:-1, 0], xticklabels=[], yticklabels=[])
    ax_out = fig.add_subplot(grid[-1, 1:], xticklabels=[], yticklabels=[])

    plot_weights_square_ax(w_input, ax_inp, plot_colorbar=True)
    plot_weights_square_ax(w_rec, ax_rec, plot_colorbar=True)
    plot_weights_square_ax(w_output, ax_out, plot_colorbar=True)

    return fig, ax_inp, ax_rec, ax_out


def plot_eigenspectra(w_rec):
    ew, ev = np.linalg.eig(w_rec)

    f, ax = plt.subplots()
    ax.scatter(ew.real, ew.imag)

    print('max real part is', ew.real.max())
    print('dot product of first two eigenvectors is',
          np.dot(ev[:, 0], ev[:, 1]))

    return f, ax


def Vderivs(V1, V2, I1, I2, w_rec, w_inp, tau=5):
    """
    Time derivatives for V variables (dV/dt).
    """

    dV1dt = -V1 + w_rec[0, 0] * V1 + w_rec[0, 1] * V2 + w_inp[0] * I1
    dV2dt = -V2 + w_rec[1, 0] * V1 + w_rec[1, 1] * V2 + w_inp[1] * I2
    return dV1dt, dV2dt


def plot_nullcline(ax, x, y, z, color='k', label=''):
    """
    Nullclines.
    """
    nc = ax.contour(x, y, z, levels=[0], colors=color)  # S1 nullcline
    nc.collections[0].set_label(label)
    return nc


def plot_flow_field(ax, x, y, dxdt, dydt, n_skip=1, scale=None, facecolor='gray'):
    """
    Vector flow fields.
    """
    v = ax.quiver(x[::n_skip, ::n_skip], y[::n_skip, ::n_skip],
                  dxdt[::n_skip, ::n_skip], dydt[::n_skip, ::n_skip],
                  angles='xy', scale_units='xy', scale=scale, facecolor=facecolor)
    return v


def plot_phase_plane(I1, I2, w_rec, w_inp, ax=None):
    """
    Phase plane plot with nullclines and flow fields.
    """

    if ax is None:
        ax = plt.gca()

    # Make 2D grid of (V1,V2) values
    V_vec = np.linspace(-1, 4, 200)  # things break down at S=0 or S=1
    V1, V2 = np.meshgrid(V_vec, V_vec)

    dV1dt, dV2dt = Vderivs(V1, V2, I1, I2, w_rec, w_inp)

    plot_nullcline(ax, V2, V1, dV1dt, color='orange', label='V1 nullcline')  # S1 nullcline
    plot_nullcline(ax, V2, V1, dV2dt, color='green', label='V2 nullcline')  # S2 nullcline
    plt.legend(loc=1)

    plot_flow_field(ax, V2, V1, dV2dt, dV1dt, n_skip=12, scale=40)

    ax.set_xlabel('$V_2$')
    ax.set_ylabel('$V_1$')
    #     ax.set_xlim(0,0.8)
    #     ax.set_ylim(0,0.8)
    ax.set_aspect('equal')
