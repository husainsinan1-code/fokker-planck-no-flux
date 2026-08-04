import numpy as np

# def stationary_solution(x, b, sigma, boundary):

#     bound_1, bound_2 = boundary

#     if b(0,0) == 0:
#         p = np.ones_like(x) / (bound_2 - bound_1)

#     else:
#         k = 2 * b(0,0) / sigma(0,0)**2
#         const = (np.exp(k * bound_2) - np.exp(k * bound_1)) / k
#         p = np.exp(k * x) / const

#     return p

def stationary_solution(x, b, sigma, boundary):

    bound_1, bound_2 = boundary

    x = np.asarray(x)

    b_value = b(0)
    sigma_1 = sigma(bound_1)
    sigma_2 = sigma(bound_2)

    # case sigma is constant
    if sigma_1 == sigma_2:

        if b_value == 0:
            p = np.ones_like(x) / (bound_2 - bound_1)

        else:
            k = 2 * b_value / sigma_1**2
            const = (np.exp(k * bound_2) - np.exp(k * bound_1)) / k
            p = np.exp(k * x) / const

    # sigma is linear
    else:

        slope = (sigma_2 - sigma_1) / (bound_2 - bound_1)
        intercept = sigma_1 - slope * bound_1

        sigma_x = slope*x + intercept
        sigma_1 = slope*bound_1 + intercept
        sigma_2 = slope*bound_2 + intercept

        if b_value == 0:
            p = 1 / sigma_x**2
            const = (1/slope) * (1/sigma_1 - 1/sigma_2)

        else:
            p = np.exp(-2*b_value/(slope*sigma_x)) / sigma_x**2
            const = (np.exp(-2*b_value/(slope*sigma_2)) 
                     - np.exp(-2*b_value/(slope*sigma_1))) / (2*b_value)

        p = p / const

    return p



def stationary_solution_2d(x, y, b, sigma, boundaries):

    #need to verify this code later
    
    x_min, x_max = boundaries[0]
    y_min, y_max = boundaries[1]

    Sigma = sigma @ sigma.T
    k = 2 * np.linalg.solve(Sigma, b)

    k1 = k[0]
    k2 = k[1]

    if np.isclose(k1, 0):
        const_x = x_max - x_min
    else:
        const_x = (np.exp(k1*x_max) - np.exp(k1*x_min)) / k1

    if np.isclose(k2, 0):
        const_y = y_max - y_min
    else:
        const_y = (np.exp(k2*y_max) - np.exp(k2*y_min)) / k2

    const = const_x * const_y

    p = np.exp(k1*x + k2*y) / const

    return p


def stationary_solution_disk(x, y, b, sigma, radius=1.0):
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

def stationary_solution_prime(x, b, sigma, boundary):

    bound_1, bound_2 = boundary


    k = 2 * b(0,0) / sigma(0,0)**2
    const = (np.exp(k * bound_2) - np.exp(k * bound_1)) / k
    p = k*np.exp(k * x) / const

    return p



def histogram_density(X, x_range, bins):
    edges = np.linspace(x_range[0], x_range[1], bins + 1)
    bin_widths = np.diff(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])

    counts, _ = np.histogram(X, bins=edges)

    density = counts / (len(X) * bin_widths)

    return density, centres, counts, edges, bin_widths

def density_errors(p_num, p_true, bin_widths, selection=["local","L1", "L2", "relative"]):

    local_error = p_num - p_true

    L1_error = np.sum(np.abs(local_error) *bin_widths)

    L2_error = np.sqrt(np.sum(local_error**2 *bin_widths))
    relative_L2_error = L2_error / np.sqrt(np.sum(p_true**2 * bin_widths))

    errors = {"local": local_error
              , "L1": L1_error
              , "L2": L2_error
              , "relative": relative_L2_error}

    return tuple(errors[name] for name in selection)



def histogram_density_disk(X, radius=1.0, bins=50, density=True):
    bounds = ((-radius, radius), (-radius, radius))
    p, x_edges, y_edges = np.histogram2d(X[:, 0], X[:, 1], bins=bins, range=bounds, density=density)

    x_centres = 0.5*(x_edges[:-1] + x_edges[1:])
    y_centres = 0.5*(y_edges[:-1] + y_edges[1:])
    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres, indexing="ij")

    p[x_mesh**2 + y_mesh**2 > radius**2] = np.nan

    return p, x_edges, y_edges


def density_disk_range(X, r_range, theta_range=(0, 2*np.pi), theta_bins=60, r_bins=20, density=True):
    x = X[:, 0]
    y = X[:, 1]

    r = np.sqrt(x**2 + y**2)
    theta = np.mod(np.arctan2(y, x), 2*np.pi)

    r_min, r_max = r_range
    theta_min, theta_max = theta_range

    index = (r >= r_min) & (r <= r_max) & (theta >= theta_min) & (theta <= theta_max)

    theta_edges = np.linspace(theta_min, theta_max, theta_bins + 1)
    r_edges = np.linspace(r_min, r_max, r_bins + 1)

    counts, theta_edges, r_edges = np.histogram2d(theta[index], r[index], bins=[theta_edges, r_edges])

    if density:
        dtheta = np.diff(theta_edges)
        area = 0.5*(r_edges[1:]**2 - r_edges[:-1]**2)
        bin_area = dtheta[:, None]*area[None, :]
        p = counts/(len(X)*bin_area)
    else:
        p = counts

    return p, theta_edges, r_edges, counts


def get_colorbar_values_2d(densities):
    vmin = min(np.nanmin(p) for p in densities)
    vmax = max(np.nanmax(p) for p in densities)

    return vmin, vmax


def local_error_density(p_num, p_ref):
    return p_num - p_ref


def density_error_norms(p_error, theta_edges, r_edges):
    dtheta = np.diff(theta_edges)
    dr_area = 0.5*(r_edges[1:]**2 - r_edges[:-1]**2)
    bin_area = dtheta[:, None]*dr_area[None, :]

    L1_error = np.nansum(np.abs(p_error)*bin_area)
    L2_error = np.sqrt(np.nansum(p_error**2*bin_area))

    return L1_error, L2_error


def density_error_norms_cartesian(p_error, x_edges, y_edges):
    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    bin_area = dx[:, None]*dy[None, :]

    L1_error = np.nansum(np.abs(p_error)*bin_area)
    L2_error = np.sqrt(np.nansum(p_error**2*bin_area))

    return L1_error, L2_error
