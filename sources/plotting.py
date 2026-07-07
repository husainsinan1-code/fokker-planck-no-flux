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