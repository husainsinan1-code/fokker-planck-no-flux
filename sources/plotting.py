import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(X, bins='fd', ax=None, title=None, x_label="x", y_label="density"
                   , density=True, x_limit=None):
    
    if ax is None:
        ax = plt.gca()

    ax.hist(X, bins=bins, density=density, label="Numerical")

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if x_limit is not None:
        ax.set_xlim(x_limit)

    ax.grid(True)

    return ax

def plot_histogram_2d(X, bins=50, ax=None, title=None, x_label="x", y_label="y", density=True
                      , limits=None, colorbar=True):
    
    if ax is None:
        ax = plt.gca()

    histo = ax.hist2d(X[:, 0], X[:, 1], bins=bins, density=density, range=limits)

    if colorbar:
        plt.colorbar(histo[3], ax=ax, label="density", fraction=0.05)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])

    ax.set_aspect("equal")
    ax.grid(True)

    return ax

def plot_bars_2d(X, bins=50, ax=None, title=None, x_label="x", y_label="y", density=True,
                          limits=None, colorbar=True):
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    histo, x_edges, y_edges = np.histogram2d(X[:, 0], X[:, 1], bins=bins, density=density, range=limits)

    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])

    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]

    X_centres, Y_centres = np.meshgrid(x_centres, y_centres, indexing="ij")

    x_pos = X_centres.ravel()
    y_pos = Y_centres.ravel()
    z_pos = np.zeros_like(x_pos)

    dx_bar = dx * np.ones_like(x_pos)
    dy_bar = dy * np.ones_like(y_pos)
    dz_bar = histo.ravel()

    ax.bar3d(x_pos, y_pos, z_pos, dx_bar, dy_bar, dz_bar, shade=True)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel("density")

    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])

    return ax



def plot_local_density(X, x_range, bins=30, ax=None, title=None, x_label="x"
                       , y_label="density"):

    if ax is None:
        ax = plt.gca()

    x_left, x_right = x_range
    edges = np.linspace(x_left, x_right, bins + 1)

    counts, _ = np.histogram(X, bins=edges)
    widths = np.diff(edges)

    density = counts / (len(X) * widths)
    centres = 0.5 * (edges[:-1] + edges[1:])

    ax.bar(centres, density, width=widths, align="center", label="Numerical")

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_left, x_right)
    ax.grid(True)

    return centres, density, edges