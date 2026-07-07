import numpy as np

def stationary_solution(x, b, sigma, boundary):

    bc_1, bc_2 = boundary

    if b == 0:
        p = np.ones_like(x) / (bc_2 - bc_1)

    else:
        k = 2 * b / sigma**2
        const = (np.exp(k * bc_2) - np.exp(k * bc_1)) / k
        p = np.exp(k * x) / const

    return p

