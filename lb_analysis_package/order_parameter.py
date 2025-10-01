import numpy as np

def calc_order_parameter(positions: np.ndarray) -> float:
    '''
    Compute the order parameter of a set of positions.

    Parameters
    ----------
    positions : np.ndarray
        The positions of the particles.

    Returns
    -------
        order_parameter : float
        The order parameter of the system.
    '''
    # Compute the order parameter
    order_parameter = 0
    q_values = 2 * np.pi / np.arange(1, 10) # this should be a 3d array to take dot product with position difference
    for q in q_values:
        old_order_parameter = order_parameter
        for j in range(len(positions)):
            for k in range(len(positions)):
                if j != k:
                    order_parameter += np.exp(-1j * q * (positions[j] - positions[k]))
        order_parameter /= len(positions)
        if order_parameter < old_order_parameter:
            order_parameter = old_order_parameter

    return order_parameter