import numpy as np

def stationary_solution(x, b, sigma, boundary):

    bound_1, bound_2 = boundary

    if b(0,0) == 0:
        p = np.ones_like(x) / (bound_2 - bound_1)

    else:
        k = 2 * b(0,0) / sigma(0,0)**2
        const = (np.exp(k * bound_2) - np.exp(k * bound_1)) / k
        p = np.exp(k * x) / const

    return p

def stationary_solution_prime(x, b, sigma, boundary):

    bound_1, bound_2 = boundary


    k = 2 * b(0,0) / sigma(0,0)**2
    const = (np.exp(k * bound_2) - np.exp(k * bound_1)) / k
    p = k*np.exp(k * x) / const

    return p

