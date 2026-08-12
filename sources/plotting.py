import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from sources.simulation import disk_boundary


# =============================================================================
# Shared plotting helpers
# =============================================================================

def add_shared_colorbar(fig, ax, colorbar_values, cmap=None, label="density",
                        fraction=0.03, pad=0.03):
    """Add one shared colorbar to a figure."""

    vmin, vmax = colorbar_values
    mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)

    if isinstance(ax, np.ndarray):
        ax = ax.ravel().tolist()

    fig.colorbar(mappable, ax=ax, label=label, fraction=fraction, pad=pad)

    return fig


# =============================================================================
# 1D plotting
# =============================================================================

def plot_histogram(X, bins="fd", ax=None, title=None, x_label="x",
                   y_label="density", density=True, x_limit=None):
    """Plot a 1D histogram from particle samples."""

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


def plot_local_density(centres, density, widths, ax=None, title=None,
                       x_label="x", y_label="density", label="Numerical"):
    """Plot a precomputed 1D local density estimate."""

    if ax is None:
        ax = plt.gca()

    ax.bar(centres, density, width=widths, align="center", label=label)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)

    return ax


# =============================================================================
# 2D Cartesian plotting
# =============================================================================

def plot_density_mesh(p, x_edges, y_edges, ax=None, title=None,
                      colorbar_values=None, cmap=None, x_label="x",
                      y_label="y", equal_aspect=False):
    """Plot a gridded 2D field using supplied mesh edges."""

    if ax is None:
        ax = plt.gca()

    vmin, vmax = (None, None) if colorbar_values is None else colorbar_values

    ax.pcolormesh(
        x_edges,
        y_edges,
        p.T,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if equal_aspect:
        ax.set_aspect("equal")

    ax.grid(False)

    return ax


def plot_density_2d(p, bounds, ax=None, title=None,
                    colorbar_values=None, cmap=None):
    """Plot a rectangular 2D field when only bounds are given."""

    x_edges = np.linspace(bounds[0][0], bounds[0][1], p.shape[0] + 1)
    y_edges = np.linspace(bounds[1][0], bounds[1][1], p.shape[1] + 1)

    plot_density_mesh(
        p,
        x_edges,
        y_edges,
        ax=ax,
        title=title,
        colorbar_values=colorbar_values,
        cmap=cmap,
        x_label="x",
        y_label="y",
        equal_aspect=True
    )

    if ax is None:
        ax = plt.gca()

    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])

    return ax


def plot_histogram_2d(X, bins=50, ax=None, title=None, x_label="x",
                      y_label="y", density=True, limits=None,
                      colorbar=False, colorbar_values=None, cmap=None):
    """Plot a 2D Cartesian histogram from particle samples."""

    if ax is None:
        ax = plt.gca()

    vmin, vmax = (None, None) if colorbar_values is None else colorbar_values

    histo = ax.hist2d(
        X[:, 0],
        X[:, 1],
        bins=bins,
        density=density,
        range=limits,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )

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
    ax.grid(False)

    return ax


def plot_bars_2d(p, x_edges, y_edges, ax=None, title=None,
                 x_label="x", y_label="y", z_label="density"):
    """Plot a 3D bar plot from a precomputed 2D density array."""

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    x_centres = 0.5*(x_edges[:-1] + x_edges[1:])
    y_centres = 0.5*(y_edges[:-1] + y_edges[1:])

    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]

    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres, indexing="ij")

    x_pos = x_mesh.ravel()
    y_pos = y_mesh.ravel()
    z_pos = np.zeros_like(x_pos)

    dx_bar = dx*np.ones_like(x_pos)
    dy_bar = dy*np.ones_like(y_pos)
    dz_bar = p.ravel()

    ax.bar3d(x_pos, y_pos, z_pos, dx_bar, dy_bar, dz_bar, shade=True)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    return ax


# =============================================================================
# 2D disk plotting
# =============================================================================

def plot_disk_boundary(ax=None, radius=1.0, n_points=300):
    """Plot the boundary of a disk."""

    if ax is None:
        ax = plt.gca()

    theta = np.linspace(0, 2*np.pi, n_points)
    x_boundary, y_boundary = disk_boundary(theta=theta, radius=radius)

    ax.plot(x_boundary, y_boundary)

    return ax


def plot_density_disk(p, x_edges, y_edges, radius=1.0, ax=None,
                      title=None, colorbar_values=None, cmap=None,
                      show_boundary=True):
    """Plot a precomputed Cartesian density field in a disk."""

    if ax is None:
        ax = plt.gca()

    plot_density_mesh(
        p,
        x_edges,
        y_edges,
        ax=ax,
        title=title,
        colorbar_values=colorbar_values,
        cmap=cmap,
        x_label="x",
        y_label="y",
        equal_aspect=True
    )

    if show_boundary:
        plot_disk_boundary(ax=ax, radius=radius)

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)

    return ax


def plot_density_disk_range(p, theta_edges, r_edges, ax=None, title=None,
                            colorbar_values=None, cmap=None):
    """Plot a precomputed polar density field in a disk annulus."""

    if ax is None:
        ax = plt.gca()

    plot_density_mesh(
        p,
        theta_edges,
        r_edges,
        ax=ax,
        title=title,
        colorbar_values=colorbar_values,
        cmap=cmap,
        x_label=r"$\theta$",
        y_label=r"$r$",
        equal_aspect=False
    )

    ax.set_xlim(theta_edges[0], theta_edges[-1])
    ax.set_ylim(r_edges[0], r_edges[-1])

    return ax


# =============================================================================
# Radial slice plotting
# =============================================================================

def plot_radial_slice(r, values, ax=None, title=None, label=None,
                      y_label="value"):
    """Plot precomputed values along a radial slice."""

    if ax is None:
        ax = plt.gca()

    ax.plot(r, values, marker="x", label=label)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel("r")
    ax.set_ylabel(y_label)
    ax.grid(True)

    if label is not None:
        ax.legend()

    return ax