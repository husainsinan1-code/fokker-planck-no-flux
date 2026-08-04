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

def plot_density_mesh(p, x_edges, y_edges, ax=None, title=None, colorbar_values=None, cmap=None, x_label="x", y_label="y", equal_aspect=False):
    if ax is None:
        ax = plt.gca()

    vmin, vmax = (None, None) if colorbar_values is None else colorbar_values

    ax.pcolormesh(x_edges, y_edges, p.T, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if equal_aspect:
        ax.set_aspect("equal")

    ax.grid(False)

    return ax


def plot_histogram_disk(X, radius=1.0, bins=50, ax=None, title=None, density=True, colorbar_values=None, cmap=None):
    if ax is None:
        ax = plt.gca()

    vmin, vmax = (None, None) if colorbar_values is None else colorbar_values

    p, x_edges, y_edges = histogram_density_disk(X, radius=radius, bins=bins, density=density)

    ax.pcolormesh(x_edges, y_edges, p.T, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)

    theta = np.linspace(0, 2*np.pi, 300)
    x_boundary, y_boundary = disk_boundary(radius=radius, theta=theta)
    ax.plot(x_boundary, y_boundary)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.grid(False)

    return ax


def plot_density_disk_range(X, r_range, theta_range=(0, 2*np.pi), theta_bins=60, r_bins=20, ax=None, title=None, density=True, colorbar_values=None, cmap=None):
    if ax is None:
        ax = plt.gca()

    vmin, vmax = (None, None) if colorbar_values is None else colorbar_values

    p, theta_edges, r_edges, counts = density_disk_range(
        X,
        r_range=r_range,
        theta_range=theta_range,
        theta_bins=theta_bins,
        r_bins=r_bins,
        density=density
    )

    ax.pcolormesh(theta_edges, r_edges, p.T, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$r$")
    ax.set_xlim(theta_range)
    ax.set_ylim(r_range)
    ax.grid(False)

    return ax
