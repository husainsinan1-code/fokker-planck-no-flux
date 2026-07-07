import numpy as np


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


def euler_maruyama(
    X_current,
    dt,
    b,
    sigma,
    boundary_type,
    boundaries,
    xi=None,
):
    """
    One Euler-Maruyama step.

    b and sigma may be constants or functions of x.
    """
    if xi is None:
        xi = np.random.normal(0, 1, size=X_current.shape)

    drift = b(X_current) if callable(b) else b
    diffusion = sigma(X_current) if callable(sigma) else sigma

    proposal = X_current + drift * dt + diffusion * np.sqrt(dt) * xi

    return apply_boundary(
        proposal=proposal,
        X_current=X_current,
        boundary_type=boundary_type,
        boundaries=boundaries,
    )


def numerical_solution(
    method,
    X_0,
    dt,
    n_steps,
    b,
    sigma,
    boundary_type,
    boundaries,
    save_every=None,
):
    """
    Simulate n_steps using the supplied numerical method.

    save_every:
        None  -> only final state is saved
        integer -> save a copy every save_every steps
    """
    X_current = X_0.copy()

    saved_steps = []
    saved_X = []

    for i in range(n_steps):
        X_current = method(
            X_current=X_current,
            dt=dt,
            b=b,
            sigma=sigma,
            boundary_type=boundary_type,
            boundaries=boundaries,
        )

        step_number = i + 1

        if save_every is not None and step_number % save_every == 0:
            saved_steps.append(step_number)
            saved_X.append(X_current.copy())

    if not saved_steps or saved_steps[-1] != n_steps:
        saved_steps.append(n_steps)
        saved_X.append(X_current.copy())

    return X_current, saved_steps, saved_X