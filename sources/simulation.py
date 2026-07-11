import numpy as np
from sources.data_storage import * 


def apply_boundary(proposal, X_current, boundary_type, boundaries):

    bc_1, bc_2 = boundaries

    if isinstance(boundary_type, tuple) and boundary_type[0] == "theta":
        theta = boundary_type[1]

        outside = (proposal < bc_1) | (proposal > bc_2)

        reflected = proposal.copy()

        left = reflected < bc_1
        right = reflected > bc_2

        while np.any(left) or np.any(right):
            reflected[left] = 2 * bc_1 - reflected[left]
            reflected[right] = 2 * bc_2 - reflected[right]

            left = reflected < bc_1
            right = reflected > bc_2

        reflect = np.random.random(proposal.shape) < theta

        proposal[outside & reflect] = reflected[outside & reflect]
        proposal[outside & ~reflect] = X_current[outside & ~reflect]

        return proposal


    elif boundary_type == "reflection":
        left = proposal < bc_1
        right = proposal > bc_2

        while np.any(left) or np.any(right):
            proposal[left] = 2 * bc_1 - proposal[left]
            proposal[right] = 2 * bc_2 - proposal[right]

            left = proposal < bc_1
            right = proposal > bc_2

        return proposal

    elif boundary_type == "rejection":
        outside = (proposal < bc_1) | (proposal > bc_2)

        proposal[outside] = X_current[outside]

        return proposal

    else:
        raise ValueError(f"{boundary_type=} is unknown")


def euler_maruyama(X_current, dt, b, sigma, boundary_type, boundaries, xi=None):
    
    
    if xi is None:
        xi = np.random.normal(0, 1, size=X_current.shape)

    # drift = b(X_current) if callable(b) else b
    # diffusion = sigma(X_current) if callable(sigma) else sigma

    proposal = X_current + b(X_current,dt) * dt + sigma(X_current, dt) * np.sqrt(dt) * xi

    return apply_boundary(proposal=proposal, X_current=X_current, boundary_type=boundary_type,
        boundaries=boundaries)


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