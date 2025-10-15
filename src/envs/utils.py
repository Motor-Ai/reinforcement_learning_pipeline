import numpy as np
from numpy.typing import NDArray


def ego_to_global(action: NDArray[np.float64], ego_position: NDArray[np.float64], ego_yaw: float):
    """
    Convert an (x,y) point in ego frame into a global coordinate.
    """
    cos_theta = np.cos(ego_yaw)
    sin_theta = np.sin(ego_yaw)
    R = np.array([[cos_theta, -sin_theta],
                    [sin_theta, cos_theta]])
    global_point = (R @ action.reshape(2, 1)).reshape(2,) + ego_position
    return global_point
