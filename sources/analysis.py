import numpy as np


# =============================================================================
# 1D stationary solutions
# =============================================================================

def stationary_solution(x, b, sigma, boundary):
    """Compute the 1D stationary density for constant drift and constant or linear diffusion."""

    bound_1, bound_2 = boundary

    x = np.asarray(x)

    b_value = b(0)
    sigma_1 = sigma(bound_1)
    sigma_2 = sigma(bound_2)

    # constant diffusion
    if sigma_1 == sigma_2:

        if b_value == 0:
            p = np.ones_like(x)/(bound_2 - bound_1)

        else:
            k = 2*b_value/sigma_1**2
            const = (np.exp(k*bound_2) - np.exp(k*bound_1))/k
            p = np.exp(k*x)/const

    # linear diffusion
    else:

        slope = (sigma_2 - sigma_1)/(bound_2 - bound_1)
        intercept = sigma_1 - slope*bound_1

        sigma_x = slope*x + intercept
        sigma_1 = slope*bound_1 + intercept
        sigma_2 = slope*bound_2 + intercept

        if b_value == 0:
            p = 1/sigma_x**2
            const = (1/slope)*(1/sigma_1 - 1/sigma_2)

        else:
            p = np.exp(-2*b_value/(slope*sigma_x))/sigma_x**2
            const = (
                np.exp(-2*b_value/(slope*sigma_2))
                - np.exp(-2*b_value/(slope*sigma_1))
            )/(2*b_value)

        p = p/const

    return p


def stationary_solution_prime(x, b, sigma, boundary):
    """Compute the derivative of the constant-diffusion 1D stationary density."""

    bound_1, bound_2 = boundary

    k = 2*b(0)/sigma(0)**2
    const = (np.exp(k*bound_2) - np.exp(k*bound_1))/k
    p_prime = k*np.exp(k*x)/const

    return p_prime


# =============================================================================
# 2D stationary solutions
# =============================================================================

def stationary_solution_2d(x, y, b, sigma, boundaries):
    """Compute the 2D stationary density for constant drift and diffusion in a rectangle."""

    x_min, x_max = boundaries[0]
    y_min, y_max = boundaries[1]

    Sigma = sigma @ sigma.T
    k = 2*np.linalg.solve(Sigma, b)

    k1 = k[0]
    k2 = k[1]

    if np.isclose(k1, 0):
        const_x = x_max - x_min
    else:
        const_x = (np.exp(k1*x_max) - np.exp(k1*x_min))/k1

    if np.isclose(k2, 0):
        const_y = y_max - y_min
    else:
        const_y = (np.exp(k2*y_max) - np.exp(k2*y_min))/k2

    const = const_x*const_y

    p = np.exp(k1*x + k2*y)/const

    return p


def stationary_solution_disk(x, y, b, sigma, radius=1.0):
    """Compute the 2D stationary density for constant drift and diffusion in a disk."""

    Sigma = sigma @ sigma.T
    k = 2*np.linalg.solve(Sigma, b)

    grid_size = 500
    x_grid = np.linspace(-radius, radius, grid_size)
    y_grid = np.linspace(-radius, radius, grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid, indexing="ij")

    p_grid = np.exp(k[0]*x_mesh + k[1]*y_mesh)
    mask_grid = x_mesh**2 + y_mesh**2 <= radius**2

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    const = np.sum(p_grid[mask_grid])*dx*dy

    p = np.exp(k[0]*x + k[1]*y)/const

    mask = x**2 + y**2 <= radius**2
    p[~mask] = np.nan

    return p


# =============================================================================
# 1D density and errors
# =============================================================================

def histogram_density(X, x_range, bins):
    """Compute a 1D histogram density from particle samples."""

    edges = np.linspace(x_range[0], x_range[1], bins + 1)
    bin_widths = np.diff(edges)
    centres = 0.5*(edges[:-1] + edges[1:])

    counts, _ = np.histogram(X, bins=edges)

    density = counts/(len(X)*bin_widths)

    return density, centres, counts, edges, bin_widths


def density_errors(p_num, p_true, bin_widths,
                   selection=("local", "L1", "L2", "relative")):
    """Compute selected 1D density error quantities."""

    local_error = p_num - p_true

    L1_error = np.sum(np.abs(local_error)*bin_widths)
    L2_error = np.sqrt(np.sum(local_error**2*bin_widths))
    relative_L2_error = L2_error/np.sqrt(np.sum(p_true**2*bin_widths))

    errors = {
        "local": local_error,
        "L1": L1_error,
        "L2": L2_error,
        "relative": relative_L2_error
    }

    return tuple(errors[name] for name in selection)


# =============================================================================
# 2D Cartesian density and errors
# =============================================================================

def histogram_density_2d(X, x_edges, y_edges, density=True):
    """Compute a 2D Cartesian histogram density from particle samples."""

    p, x_edges, y_edges = np.histogram2d(
        X[:, 0],
        X[:, 1],
        bins=[x_edges, y_edges],
        density=density
    )

    return p, x_edges, y_edges


def histogram_density_disk(X, radius=1.0, bins=50, density=True):
    """Compute a Cartesian histogram density from particle samples in a disk."""

    bounds = ((-radius, radius), (-radius, radius))

    p, x_edges, y_edges = np.histogram2d(
        X[:, 0],
        X[:, 1],
        bins=bins,
        range=bounds,
        density=density
    )

    x_centres = 0.5*(x_edges[:-1] + x_edges[1:])
    y_centres = 0.5*(y_edges[:-1] + y_edges[1:])
    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres, indexing="ij")

    p[x_mesh**2 + y_mesh**2 > radius**2] = np.nan

    return p, x_edges, y_edges


def local_error_density(p_num, p_ref):
    """Compute pointwise density error."""

    return p_num - p_ref


def density_error_norms_cartesian(p_error, p_ref, x_edges, y_edges):
    """Compute L1, L2, and relative L2 errors for a Cartesian density grid."""

    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    bin_area = dx[:, None]*dy[None, :]

    L1_error = np.nansum(np.abs(p_error)*bin_area)
    L2_error = np.sqrt(np.nansum(p_error**2*bin_area))

    reference_norm = np.sqrt(np.nansum(p_ref**2*bin_area))
    relative_L2_error = L2_error/reference_norm

    return L1_error, L2_error, relative_L2_error


# =============================================================================
# 2D disk polar density and errors
# =============================================================================

def density_disk_range(X, r_range, theta_range=(0, 2*np.pi),
                       theta_bins=60, r_bins=20, density=True):
    """Compute polar density from particle samples in a disk annulus."""

    x = X[:, 0]
    y = X[:, 1]

    r = np.sqrt(x**2 + y**2)
    theta = np.mod(np.arctan2(y, x), 2*np.pi)

    r_min, r_max = r_range
    theta_min, theta_max = theta_range

    index = (
        (r >= r_min)
        & (r <= r_max)
        & (theta >= theta_min)
        & (theta <= theta_max)
    )

    theta_edges = np.linspace(theta_min, theta_max, theta_bins + 1)
    r_edges = np.linspace(r_min, r_max, r_bins + 1)

    counts, theta_edges, r_edges = np.histogram2d(
        theta[index],
        r[index],
        bins=[theta_edges, r_edges]
    )

    if density:
        dtheta = np.diff(theta_edges)
        dr_area = 0.5*(r_edges[1:]**2 - r_edges[:-1]**2)
        bin_area = dtheta[:, None]*dr_area[None, :]
        p = counts/(len(X)*bin_area)

    else:
        p = counts

    return p, theta_edges, r_edges, counts


def density_error_norms(p_error, p_ref, theta_edges, r_edges):
    """Compute L1, L2, and relative L2 errors for a polar density grid."""

    dtheta = np.diff(theta_edges)
    dr_area = 0.5*(r_edges[1:]**2 - r_edges[:-1]**2)
    bin_area = dtheta[:, None]*dr_area[None, :]

    L1_error = np.nansum(np.abs(p_error)*bin_area)
    L2_error = np.sqrt(np.nansum(p_error**2*bin_area))

    reference_norm = np.sqrt(np.nansum(p_ref**2*bin_area))
    relative_L2_error = L2_error/reference_norm

    return L1_error, L2_error, relative_L2_error


# =============================================================================
# Radial slices
# =============================================================================

def density_disk_radial_slice(X, theta_0, radius=1.0,
                              theta_width=0.1, r_bins=40):
    """Compute particle density along a fixed-angle radial slice in the disk."""

    x = X[:, 0]
    y = X[:, 1]

    r = np.sqrt(x**2 + y**2)
    theta = np.mod(np.arctan2(y, x), 2*np.pi)

    theta_0 = np.mod(theta_0, 2*np.pi)

    angle_diff = np.angle(np.exp(1j*(theta - theta_0)))
    index = (np.abs(angle_diff) <= theta_width/2) & (r <= radius)

    r_edges = np.linspace(0, radius, r_bins + 1)
    counts, _ = np.histogram(r[index], bins=r_edges)

    r_centres = 0.5*(r_edges[:-1] + r_edges[1:])

    sector_area = 0.5*theta_width*(r_edges[1:]**2 - r_edges[:-1]**2)
    density = counts/(len(X)*sector_area)

    return r_centres, density, counts


def field_disk_radial_slice(field, x_edges, y_edges, theta_0,
                            radius=1.0, r_bins=40):
    """Extract nearest-bin field values along a fixed-angle radial slice."""

    theta_0 = np.mod(theta_0, 2*np.pi)

    r_centres = np.linspace(0, radius, r_bins)
    x_slice = r_centres*np.cos(theta_0)
    y_slice = r_centres*np.sin(theta_0)

    x_centres = 0.5*(x_edges[:-1] + x_edges[1:])
    y_centres = 0.5*(y_edges[:-1] + y_edges[1:])

    x_index = np.searchsorted(x_centres, x_slice)
    y_index = np.searchsorted(y_centres, y_slice)

    x_index = np.clip(x_index, 0, len(x_centres) - 1)
    y_index = np.clip(y_index, 0, len(y_centres) - 1)

    values = field[x_index, y_index]

    return r_centres, values


# =============================================================================
# Shared analysis helpers
# =============================================================================

def get_colorbar_values_2d(densities):
    """Return common minimum and maximum values for a list of 2D arrays."""

    vmin = min(np.nanmin(p) for p in densities)
    vmax = max(np.nanmax(p) for p in densities)

    return vmin, vmax