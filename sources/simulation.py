import numpy as np
from sources.data_storage import * 


def apply_boundary(proposal, X_current, boundary_type, boundaries):

    x_min, x_max = boundaries

    if isinstance(boundary_type, tuple) and boundary_type[0] == "theta":
        theta = boundary_type[1]

        outside = (proposal < x_min) | (proposal > x_max)

        reflected = proposal.copy()

        left = reflected < x_min
        right = reflected > x_max

        while np.any(left) or np.any(right):
            reflected[left] = 2 * x_min - reflected[left]
            reflected[right] = 2 * x_max - reflected[right]

            left = reflected < x_min
            right = reflected > x_max

        reflect = np.random.random(proposal.shape) < theta

        proposal[outside & reflect] = reflected[outside & reflect]
        proposal[outside & ~reflect] = X_current[outside & ~reflect]

        return proposal


    elif boundary_type == "reflection":
        left = proposal < x_min
        right = proposal > x_max

        while np.any(left) or np.any(right):
            proposal[left] = 2 * x_min - proposal[left]
            proposal[right] = 2 * x_max - proposal[right]

            left = proposal < x_min
            right = proposal > x_max

        return proposal

    elif boundary_type == "rejection":
        outside = (proposal < x_min) | (proposal > x_max)

        proposal[outside] = X_current[outside]

        return proposal

    else:
        raise ValueError(f"{boundary_type=} is unknown")


def apply_boundary_2d(proposal, X_current, boundary_type, boundaries, sigma):

    x_min, x_max = boundaries[0]
    y_min, y_max = boundaries[1]

    if boundary_type == "reflection":

        left = proposal[:, 0] < x_min
        right = proposal[:, 0] > x_max
        bottom = proposal[:, 1] < y_min
        top = proposal[:, 1] > y_max

        while np.any(left) or np.any(right) or np.any(bottom) or np.any(top):

            if np.any(left):
                n = np.array([-1.0, 0.0])
                v = sigma @ n
                v = v / np.linalg.norm(v)
                proposal[left, :] = proposal[left, :] - 2*((proposal[left, 0] - x_min)/v[0])[:, None]*v

            if np.any(right):
                n = np.array([1.0, 0.0])
                v = sigma @ n
                v = v / np.linalg.norm(v)
                proposal[right, :] = proposal[right, :] - 2*((proposal[right, 0] - x_max)/v[0])[:, None]*v

            if np.any(bottom):
                n = np.array([0.0, -1.0])
                v = sigma @ n
                v = v / np.linalg.norm(v)
                proposal[bottom, :] = proposal[bottom, :] - 2*((proposal[bottom, 1] - y_min)/v[1])[:, None]*v

            if np.any(top):
                n = np.array([0.0, 1.0])
                v = sigma @ n
                v = v / np.linalg.norm(v)
                proposal[top, :] = proposal[top, :] - 2*((proposal[top, 1] - y_max)/v[1])[:, None]*v

            left = proposal[:, 0] < x_min
            right = proposal[:, 0] > x_max
            bottom = proposal[:, 1] < y_min
            top = proposal[:, 1] > y_max

        return proposal

    elif boundary_type == "rejection":

        outside = ((proposal[:, 0] < x_min) | (proposal[:, 0] > x_max) |
                   (proposal[:, 1] < y_min) | (proposal[:, 1] > y_max))

        proposal[outside, :] = X_current[outside, :]

        return proposal

    else:
        raise ValueError(f"{boundary_type=} is unknown")


def euler_maruyama(X_current, dt, b, sigma, boundary_type, boundaries, xi=None):
    
    
    if xi is None:
        xi = np.random.normal(0, 1, size=X_current.shape)

    proposal = X_current + b(X_current,dt) * dt + sigma(X_current, dt) * np.sqrt(dt) * xi

    return apply_boundary(proposal=proposal, X_current=X_current, boundary_type=boundary_type,
        boundaries=boundaries)

def euler_maruyama_2d(X_current, dt, b, sigma, boundary_type, boundaries, xi=None):

    if xi is None:
        xi = np.random.normal(0, 1, size=X_current.shape)

    proposal = X_current + b * dt + np.sqrt(dt) * (xi @ sigma.T)

    return apply_boundary_2d(proposal, X_current, boundary_type, boundaries, sigma)



def numerical_solution(method, X_0, dt, n_steps, b, sigma, boundary_type, boundaries
                       , save_sim=True):
    
    
    X_current = X_0.copy()

    for i in range(n_steps):
        X_current = method(X_current=X_current, dt=dt, b=b, sigma=sigma
                           , boundary_type=boundary_type, boundaries=boundaries)


    if save_sim:
        
        filename = make_filename(boundary_type[0:3],b=b(0,0), sigma=sigma(0,0), dt=dt
                                 , n_parts=X_current.size)
        save_solution(X_current, filename)


    return X_current