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

def stationary_solution_prime(x, b, sigma, boundary):

    bound_1, bound_2 = boundary


    k = 2 * b(0,0) / sigma(0,0)**2
    const = (np.exp(k * bound_2) - np.exp(k * bound_1)) / k
    p = k*np.exp(k * x) / const

    return p


import numpy as np


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

