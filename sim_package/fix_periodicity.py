from numba import njit
import numpy as np

@njit
def fix_pbc(lipid_heads: np.ndarray, box_vectors: np.ndarray):
    '''
    Adjust lipid positions to account for periodic boundary conditions.

    Parameters
    ----------
    lipid_heads : np.ndarray
        Positions of lipid heads in the system before PBC adjustment.
    box_vectors : np.ndarray
        Box vectors of the system. These start at 0.
    '''
    x_val = box_vectors[0][0]
    y_val = box_vectors[1][1]
    z_val = box_vectors[2][2]

    for i in range(lipid_heads.shape[0]):
        if lipid_heads[i, 0] > x_val:
            lipid_heads[i, 0] -= x_val
        elif lipid_heads[i, 0] < 0:
            lipid_heads[i, 0] += x_val
        if lipid_heads[i, 1] > y_val:
            lipid_heads[i,1] -= y_val
        elif lipid_heads[i, 1] < 0:
            lipid_heads[i, 1] += y_val
        if lipid_heads[i, 2] > z_val:
            lipid_heads[i, 2] -= z_val
        elif lipid_heads[i, 2] < 0:
            lipid_heads[i, 2] += z_val
    
    return lipid_heads